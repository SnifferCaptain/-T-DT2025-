#include "carman.hpp"
#include <opencv2/opencv.hpp>

Carman::Carman() {}

Carman::Carman(float t, int persicesion, int measurementDim, int controlDim,float confP) : timeStep(t), ioDim(measurementDim), stateDim((1 + persicesion) * measurementDim),confidentTrans(confP)
{
    //persicesion是泰勒展开的阶数。
    this->controlDim=controlDim;
    kf.reset(new cv::KalmanFilter(stateDim, measurementDim, controlDim));
    // 初始化过程继续...
    cv::Mat state(stateDim * measurementDim, 1, CV_32F);
    state.setTo(cv::Scalar(0)); // 初始化状态向量为0
    cv::setIdentity(kf->transitionMatrix, cv::Scalar(1));// 设置状态转移矩阵为单位矩阵
    kf->measurementMatrix = cv::Mat::zeros(measurementDim, stateDim, CV_32F); // 测量矩阵
    cv::setIdentity(kf->measurementMatrix, cv::Scalar(1));

    if(persicesion>10)std::cout<<"oops! persicesion too big"<<std::endl;
    createTransision(t);
    cv::setIdentity(kf->processNoiseCov, cv::Scalar::all(0.05f));
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(confidentTrans));
    cv::setIdentity(kf->errorCovPost, cv::Scalar::all(10));
    cv::setIdentity(kf->controlMatrix,cv::Scalar::all(0.001f));
    kf->statePost=cv::Mat::zeros(stateDim, 1, CV_32F);
}

void Carman::createTransision(float t){
    int ds = stateDim / ioDim;
    cv::Mat transision(cv::Mat::zeros(stateDim, stateDim, CV_32F));
    cv::setIdentity(transision, cv::Scalar(1));// 设置状态转移矩阵为单位矩阵
    for (int a = 0; a < ds; a++){
        for (int b = 0; b < ioDim; b++){
            int mdf = b + ioDim * a;//当前的列
            for(int c = 0; c < a; c++){
                transision.at<float>(c*ioDim+b, mdf) = powf(t,a-c)*(1.f/stepMul(a-c));
            }
        }
    }
    kf->transitionMatrix = transision;
}

cv::Mat Carman::getTransision(float t)const{
    int ds = stateDim / ioDim;
    cv::Mat transision(cv::Mat::zeros(stateDim, stateDim, CV_32F));
    cv::setIdentity(transision, cv::Scalar(1)); // 设置状态转移矩阵为单位矩阵
    for (int a = 0; a < ds; a++)
    {
        for (int b = 0; b < ioDim; b++)
        {
            int mdf = b + ioDim * a; // 当前的列
            for (int c = 0; c < a; c++)
            {
                transision.at<float>(c * ioDim + b, mdf) = powf(t, a - c) * (1.f / stepMul(a - c));
            }
        }
    }
    return transision;
}

double Carman::stepMul(int n)const
{
    double ret = n;
    while(--n){
        ret = ret * n;
    }
    return ret;
}

float Carman::conf2noise(float conf){
    return confidentTrans/conf;
}

std::vector<float> Carman::predict(float time, std::vector<float> control) const{
    cv::Mat statePostBkup = kf->statePost;
    cv::Mat errorCovBkup = kf->errorCovPost;
    cv::Mat transitionBkup = kf->transitionMatrix;
    cv::Mat newTrans = getTransision(time);
    kf->transitionMatrix = newTrans;
    if(control.size()==controlDim){
        //启用
        kf->controlMatrix = cv::Mat::zeros(stateDim, controlDim, CV_32F);
        for (int i = 0; i < controlDim; i++){
            kf->controlMatrix.at<float>(i, i) = control[i];
        }
    }
    
    std::vector<float> ret(stateDim);
    cv::Mat prediction = kf->predict();
    for (int i = 0; i < stateDim; i++){
        ret[i] = prediction.at<float>(i);
    }
    kf->statePost = statePostBkup;
    kf->errorCovPost = errorCovBkup;
    kf->transitionMatrix = transitionBkup;
    kf->controlMatrix = cv::Mat::zeros(stateDim, controlDim, CV_32F);
    return ret;
}

std::vector<float> Carman::predict(float time) const{
    cv::Mat statePostBkup=kf->statePost;
    cv::Mat errorCovBkup=kf->errorCovPost;
    cv::Mat transitionBkup=kf->transitionMatrix;
    cv::Mat newTrans = getTransision(time);
    kf->transitionMatrix = newTrans;
    std::vector<float> ret(stateDim);
    cv::Mat prediction = kf->predict();
    for (int i = 0; i < stateDim; i++){
        ret[i] = prediction.at<float>(i);
    }
    kf->statePost = statePostBkup;
    kf->errorCovPost = errorCovBkup;
    kf->transitionMatrix = transitionBkup;    
    return ret;
}

std::vector<float> Carman::predict() const{
    cv::Mat statePostBkup=kf->statePost;
    cv::Mat errorCovBkup=kf->errorCovPost;
    cv::Mat transitionBkup=kf->transitionMatrix;
    cv::Mat newTrans = getTransision(timeStep);
    kf->transitionMatrix = newTrans;
    std::vector<float> ret(stateDim);
    cv::Mat prediction = kf->predict();
    for (int i = 0; i < stateDim; i++){
        ret[i] = prediction.at<float>(i);
    }
    kf->statePost = statePostBkup;
    kf->errorCovPost = errorCovBkup;
    kf->transitionMatrix = transitionBkup;    
    return ret;
}

std::vector<float> Carman::correct(std::vector<float> measurement, std::vector<float> control, float conf,float time){
    time=time==0.f?timeStep:time;
    cv::Mat measurementMat(measurement.size(), 1, CV_32F, measurement.data());
    createTransision(time); // 更正状态转移矩阵
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(conf2noise(conf)));
    if(control.size()==controlDim){
        //启用
        kf->controlMatrix = cv::Mat::zeros(stateDim, controlDim, CV_32F);
        for (int i = 0; i < controlDim; i++){
            kf->controlMatrix.at<float>(i, i) = control[i];
        }
    }
    kf->predict();
    cv::Mat corrected = kf->correct(measurementMat);
    std::vector<float> ret(stateDim);
    for (int i = 0; i < stateDim; i++){
        ret[i] = corrected.at<float>(i);
    }
    kf->controlMatrix = cv::Mat::zeros(stateDim, controlDim, CV_32F);
    return ret;
}

std::vector<float> Carman::correct(std::vector<float> measurement, float conf,float time){
    time=time==0.f?timeStep:time;
    cv::Mat measurementMat(measurement.size(), 1, CV_32F, measurement.data());
    createTransision(time); // 更正状态转移矩阵
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(conf2noise(conf)));
    kf->predict();
    cv::Mat corrected = kf->correct(measurementMat);
    std::vector<float> ret(stateDim);
    for (int i = 0; i < stateDim; i++){
        ret[i] = corrected.at<float>(i);
    }
    return ret;
}

std::vector<float> Carman::correct(float time){
    time = time == 0.f ? timeStep : time;
    createTransision(time); // 更正状态转移矩阵
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(conf2noise(0.1)));
    cv::Mat measurementMat;
    kf->predict().copyTo(measurementMat);
    measurementMat=measurementMat.rowRange(0,ioDim);
    cv::Mat corrected = kf->correct(measurementMat);
    std::vector<float> ret(stateDim);//当心出错
    for (int i = 0; i < stateDim; i++){
        ret[i] = corrected.at<float>(i);
    }
    return ret;
}

void Carman::initialize(std::vector<float> firstMeasurement){
    for(int i=0;i<firstMeasurement.size();i++){
        kf->statePost.at<float>(i)=firstMeasurement[i];
    }
}