#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>

class Carman
{
private:
    std::unique_ptr<cv::KalmanFilter> kf;
    float timeStep;//基础时间间隔（毫秒！）
    float confidentTrans;//置信度为1的时候，噪声的值。置信度为0.333时，为3倍噪声
    int ioDim;
    int stateDim;
    int controlDim;

    void createTransision(float timeDuration);
    cv::Mat getTransision(float timeDuration)const;
    double stepMul(int n)const;
    float conf2noise(float conf);// 为正数即可
public:
    Carman();
    Carman(float ms,int persicesion=2, int measurementDim=2, int controlDim=0,float confP=0.001f); // 输入时间间隔（单位：毫秒！）,KalmanFilterの维度を設定，测量维度，控制维度，置信度转换噪声系数(越大)
    void initialize(std::vector<float> firstMeasurement);//初始化第一次测量值&相机位置
    std::vector<float> predict(float time,std::vector<float> control)const;// 返回t毫秒后的预测值，包含像素移动信息
    std::vector<float> predict(float time)const;// 返回t毫秒后的预测值
    std::vector<float> predict() const;// 返回默认一帧后的预测值
    std::vector<float> correct(std::vector<float> measurement, std::vector<float> control, float conf = 1, float time = 0); // 包含相机位移信息
    std::vector<float> correct(std::vector<float> measurement,float conf=1,float time=0); //包含置信度与时间间隔
    std::vector<float> correct(float time=0.f);
};
