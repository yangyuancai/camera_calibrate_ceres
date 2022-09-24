#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "Eigen/Eigen"

#define __MYDEBUG__ 1

#ifdef __MYDEBUG__
	#include "opencv2/opencv.hpp"
	#include <fstream>
	#include <string>
	#include <io.h>
#endif



struct MonoReprojectionError {

	MonoReprojectionError(double x_,double y_, double z_, double u_,double v_, int image_id_, int image_num_):
		x(x_),y(y_),z(z_),u(u_),v(v_),image_id(image_id_),image_num(image_num_){}
	
	template <typename T>
	bool operator()(const T* const camera,T* residuals) const
	{
		T fx = camera[6 * image_num];		T fy = camera[6 * image_num+1];
		T cx = camera[6 * image_num+2];	T cy = camera[6 * image_num+3];
		T k1 = camera[6 * image_num+4];	T k2 = camera[6 * image_num+5];
		T p1 = camera[6 * image_num+6];	T p2 = camera[6 * image_num+7];
		T k3 = camera[6 * image_num+8];
		const T *rtvec = &camera[6 * image_id];
		T rt_point3d[3];
		const T obj_point3d[3] = { T(x),T(y),T(z) };
		ceres::AngleAxisRotatePoint(&camera[6 * image_id], obj_point3d, rt_point3d);

		rt_point3d[0] = rt_point3d[0] + rtvec[3];
		rt_point3d[1] = rt_point3d[1] + rtvec[4];
		rt_point3d[2] = rt_point3d[2] + rtvec[5];
		
		T x_norm = rt_point3d[0] / rt_point3d[2];
		T y_norm = rt_point3d[1] / rt_point3d[2];
		T r2 = x_norm * x_norm + y_norm * y_norm;
		T r_d = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
		T x_d = x_norm * r_d + T(2) * p1*x_norm*y_norm + p2 * (r2 + T(2) * x_norm*x_norm);
		T y_d = y_norm * r_d + T(2) * p2*x_norm*y_norm + p1 * (r2 + T(2) * y_norm*y_norm);

		residuals[0] = (x_d * fx + cx) - u;
		residuals[1] = (y_d * fy + cy) - v;
		#ifdef __MYDEBUG__
			std::ofstream file;
			file.open("../../data/residual.csv", std::ios::app);
			file<< "residuals = " << residuals[0] << std::endl;
			file<< "residuals = " << residuals[1] << std::endl;
			file.close();
		#endif
		return true;
	}

	static ceres::CostFunction* Create(double x_, double y_, double z_, double u_, double v_, int image_id_, int image_num_)
	{
		return (new ceres::AutoDiffCostFunction<MonoReprojectionError, 2, 6*5+9>(
				new MonoReprojectionError(x_,y_,z_,u_,v_,image_id_,image_num_)));
	}
	double x, y, z;
	double u,v;
	int image_id;
	int image_num=5;
};

#if 1
int main(int argc, char** argv)
{
	/*
	���̸�size��9*6���궨ͼ��������5
	/*
	�������꣺image_points.raw���ݸ�ʽN*3; N:����������9*6*5��3:point.x,point.y,image_id
	�������꣺obj_points.raw���ݸ�ʽN*3��N:����������9*6*5��3:x,y,z
	��ʼ��������camera.raw���ݸ�ʽ��6*N+9 6:��תʸ��+ƽ��ʸ����N �궨ͼ��������
	9���ڲ�fx,fy,cx,cy,k1,k2,p1,p2,k3
	*/
	
	FILE *fpsrc = fopen("../../data/image_points.raw", "rb");  //�����ļ�ָ��
	double *image_points = new double[270*3];  //����
	if (NULL != fpsrc)
	{
		fread(image_points, sizeof(double), 270 * 3, fpsrc);
	}
	fclose(fpsrc);

	fpsrc = fopen("../../data/obj_points.raw", "rb");  //�����ļ�ָ��
	double *obj_points = new double[270 * 3];  //����
	if (NULL != fpsrc)
	{
		fread(obj_points, sizeof(double), 270 * 3, fpsrc);
	}
	fclose(fpsrc);

	fpsrc = fopen("../../data/camera.raw", "rb");  //�����ļ�ָ��
	double *camera = new double[270 * 3];  //����
	if (NULL != fpsrc)
	{
		fread(camera, sizeof(double), 6*5+9, fpsrc);
	}
	fclose(fpsrc);
	google::InitGoogleLogging(argv[0]);
	ceres::Problem problem;
	for (int i = 0; i < 270; ++i)
	{
		ceres::CostFunction *cost_function = MonoReprojectionError::Create(	obj_points[3 * i],
																			obj_points[3 * i + 1],
																			obj_points[3 * i + 2],
																			image_points[3 * i],
																			image_points[3 * i + 1],
																			int(image_points[3 * i + 2]), 5);
		problem.AddResidualBlock(cost_function, nullptr, camera);
	}
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 10;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";
	std::cout << summary.BriefReport() << "\n";

	#ifdef __MYDEBUG__
		cv::Mat img_p = cv::Mat(270, 3, CV_64FC1, image_points);
		cv::Mat obj_p = cv::Mat(270, 3, CV_64FC1, obj_points);
		cv::Mat camera_mat = cv::Mat(39, 1, CV_64FC1, camera);
		std::cout << "rms = "<<sqrt(summary.final_cost * 2.0 / 270)<<"\n";
		std::cout << "camera = " << camera_mat << "\n";
	#endif 


	return 0;

}
#endif






#if 0
int main(int argc, char** argv)
{



	google::InitGoogleLogging(argv[0]);
	BALProblem bal_problem;
	if (!bal_problem.LoadFile("../../data/problem-16-22106-pre.txt"))
	{
		std::cerr<<"Error: unable to open file"<<std::endl;
		return -1;
	}

	const double* img_points_ptr = bal_problem.observations();
	ceres::Problem problem;

	for (int i=0; i<bal_problem.num_observations();++i)
	{
		ceres::CostFunction *cost_function = SnavelyReprojectionError::Create(img_points_ptr[2 * i], img_points_ptr[2 * i]);
		problem.AddResidualBlock(cost_function,nullptr,
								bal_problem.mutable_camera_for_observation(i),
								bal_problem.mutable_point_for_observation(i));
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = 500;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << summary.BriefReport() << "\n";


	return 0;

}

#endif

























#if 0
int main()
{
	std::string idx;
	for (size_t mem = 0; mem < 1; mem++)
	{

		auto start = std::chrono::steady_clock::now();

		std::cout << FourInOneCalibrationSDKV2Version() << std::endl;
		//�ڲα궨
		const char* left_mono_image_path;
		const char* right_mono_image_path;
		const char* left_mono_result_path;
		const char* right_mono_result_path;
		const char* tof_result_path;

		std::string left_mono_image_path_string;
		std::string right_mono_image_path_string;
		std::string left_mono_result_path_string;
		std::string right_mono_result_path_string;
		std::string tof_result_path_string;
		//��α궨
		const char* left_stereo_image_path;
		const char* right_stereo_image_path;
		const char* stereo_result_path;

		std::string left_stereo_image_path_string;
		std::string right_stereo_image_path_string;
		std::string stereo_result_path_string;

		//�����ļ�
		const char* calibration_config_path;
		const char* algorithm_config_path;
		//�����ļ�·��
		//calibration_config_path = "../../config/0603/calibration_config.ini";
		//algorithm_config_path = "../../config/0603/algorithm_config.ini";
		calibration_config_path = "../../config/0685/calibration_config.ini";
		algorithm_config_path = "../../config/0685/algorithm_config.ini";


		////��Ŀͼ��	
		//right_mono_image_path_string = "../../data/0603_300mm/0603002565000X/";
		//right_mono_result_path_string = "../../data/0603002565000X.ini";
		right_mono_image_path_string = "../../data/1/";
		right_mono_result_path_string = "../../data/1.ini";
		right_mono_image_path = right_mono_image_path_string.data();
		right_mono_result_path = right_mono_result_path_string.data();
	
		//��Ŀ����궨
		int status = -1;
		status = FourInOneCalibrationMonoCamera(right_mono_image_path, calibration_config_path,
												algorithm_config_path, right_mono_result_path, false);
		if (status != 0)
		{
			std::cout << "err :" << status << std::endl;
			std::cout << "calibration err." << std::endl;
		}
		auto end_center = std::chrono::steady_clock::now();
		std::chrono::duration<double, std::milli> time_center = end_center - start;
		std::cout << ">>>>>>>>>>>>>>>>  time: " << time_center.count() << "ms" << std::endl;
		std::cout << "finished" << std::endl;
	}
	std::getchar();

	return 0;
}

#endif







#if 0
int main(int argc, char** argv)
{	
	std::cout << FourInOneCalibrationSDKV2Version() << std::endl;	

	//��Ŀ�궨������ͼƬ·�����궨���·��
	const char* mono_image_path;
	const char* mono_result_path;
	
	//˫Ŀ�궨����������ĿͼƬ·��������Ŀ�궨�����˫Ŀ�궨������
	const char* left_stereo_image_path;
	const char* right_stereo_image_path;
	const char* left_result_path;
	const char* right_result_path;
	const char* stereo_result_path;

	//�궨�����ļ����㷨�����ļ�·��
	const char* calibration_config_path;
	const char* algorithm_config_path;
	
	if (8 != (int)argc && 6 != (int)argc)
	{
		std::cout << "argc err." << std::endl;
		return -1;
	}
	//---------------------------------��Ŀ����궨---------------------------//
	if (6==argc)
	{
		mono_image_path = argv[1];
		calibration_config_path = argv[2];
		algorithm_config_path = argv[3];
		mono_result_path = argv[4];
		std::string left("true");
		std::string right("false");
		if (0 != left.compare(argv[5]) && 0 != right.compare(argv[5]))
		{
			std::cout << "argc err." << std::endl;
			return -1;
		}
		bool isleft = (0==left.compare(argv[5])) ? true : false;
		int status = FourInOneCalibrationMonoCamera(mono_image_path, calibration_config_path,
													algorithm_config_path, mono_result_path, isleft);
		if (status != 0)
		{
			std::cout << "err :" << status << std::endl;
			std::cout << "mono calibration err." << std::endl;
		}
		else
		{
			std::cout << "mono calibration succeed." << std::endl;
		}

	}
	//-----------------------------˫Ŀ����궨-------------------------------//
	if(8==argc)
	{
		left_stereo_image_path = argv[1]; 
		right_stereo_image_path = argv[2];
		calibration_config_path = argv[3];
		algorithm_config_path = argv[4];
		left_result_path = argv[5];
		right_result_path = argv[6];
		stereo_result_path = argv[7];
		int status = FourInOneCalibrationStereoCamera(left_stereo_image_path, right_stereo_image_path,
													  calibration_config_path, algorithm_config_path,
													  left_result_path, right_result_path, stereo_result_path);
		if (status != 0)
		{
			std::cout << "err :" << status << std::endl;
			std::cout << "stero calibration err." << std::endl;
		}
		else
		{
			std::cout << "stero calibration succeed." << std::endl;
		}
	}
	return 0;
}

#endif