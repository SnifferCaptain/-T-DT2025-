#pragma once
#include<vector>
#include<string>
#include<openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include<memory>

/*
TODO: 使用preposeprocess，将输入输出集成，这样就可以使用连续内存拷贝猛猛加速
*/
#ifndef INFERCORE_H
#define INFERCORE_H


//推理的结果，全部都是归一化数据
struct Result{
    float x;
    float y;
    float w;
    float h;
    float conf;
    int class_id;
    cv::Rect toCvRect(int imgh,int imgw);
    void draw(cv::Mat& img);
};

class InferCore{
public:
    InferCore();
    InferCore(std::string model,bool optimize=true,std::string idDetect="");
    bool load(std::string model, bool optimize=true,std::string idDetect="");
    std::vector<Result> infer(cv::Mat& img);
    int inferId(cv::Mat& subimg);//bgr的subimage，bbox或者rect后的
    void postResult(cv::Mat& img,std::vector<Result>& result,bool show=false);
    std::vector<cv::Scalar> colors;
    float confThs;
    float nmsThs;
    bool avaliable;
    bool idAvaliable;//是否有id检测
private:
    int inputh,inputw,inputc,outputh,outputw;
    void nms(std::vector<Result>& results, float iou);
    std::unique_ptr<ov::Core> core;
    ov::CompiledModel model;
    ov::InferRequest request;
    ov::CompiledModel idmodel;// 数字检测
    ov::InferRequest idrequest;//tensor name: input output
    cv::Mat fillContour(cv::Mat img);
    void decodeResult(cv::Mat& img,std::vector<Result>& results);
    void fillTensor(cv::Mat& img);
    std::vector<Result> afterProcess();
    void optimizeCore();
    //algorithm
    template<typename T>
    inline T& valueAt(T* data,int x,int y,int colSize);
    
};

inline float getIou(Result& a,Result& b);

#endif