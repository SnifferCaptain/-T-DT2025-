#pragma once
#include"InferCore.h"
#include"Armor.h"

/*
TODO：
1、有时间可以使用rnn简单代替kf&决策。infer后直接端到端获取打击目标置信度。
   limit：视频数据集超级难做 
*/
#ifndef DETECTOR_H
#define DETECTOR_H



// 摄像头控制标准接口
struct CamControl{
    float deltaYaw;// 摄像头横向，单位：度
    float deltaPitch;// 摄像头上下，单位：度
    bool haveTarget;// 是否有目标
    bool dead;// 目标已经死了
    // 车移动速度，计划是绕着目标转，暂时没有启用
    float deltaH;// 车横向速度
    float deltaForward;// 根据yaw移动
    CamControl operator=(const CamControl other){
        deltaYaw=other.deltaYaw;
        deltaPitch=other.deltaPitch;
        haveTarget=other.haveTarget;
        dead=other.dead;
        deltaH=other.deltaH;
        deltaForward=other.deltaForward;
        return *this;
    }
};


class Detector{
public:
    double lightAreaThs;
    std::vector<Armor> armors;// 维护同一个装甲板的时间维度识别（vector的效率暂时最高）
    bool useFast;// 是否使用快速检测，使用speed()或者accuracy()函数 
    bool useZero;// 使用零匹配机制，专门打窜得快的
    float minimumIouThs;// 匹配继承IOU最小阈值
    bool judgeId;// 是否使用id判断
    float liveRate;// 存活率，0.25分界
    float realDeltaX;// 相机移动真实像素值（用于kf）
    float realDeltaY;// 相机移动真实像素值（用于kf）
    int cd;// 冷却时间，等视觉中心移动到目标中心时，才开始锁死
    
    Detector();
    Detector(std::string model_path, bool optimize = true, std::string NumDetectLoc="");
    bool loadModel(std::string model_path,bool optimize=true);//加载模型
    void speed();// 快速检测，但是会降低精度
    void accuracy();// 更加精确，但是会有更高的性能开销
    cv::Mat getSubImage(cv::Mat& img,Result& result)const;//获取检测框的图像
    inline std::vector<Result> detect(cv::Mat& img);// 检测图像
    cv::Rect getRect(cv::Mat& img,Result& result)const;// 获取检测结果的矩形框
    // 正上方开始，逆时针获取灯条四角，全局坐标
    //如果返回为空，说明不能用于解算pnp，使用infer的xywh坐标计算
    std::vector<cv::Point> getLightPoints(cv::Mat&img,Result& result);
    std::vector<cv::Point>& setCorrectDirection(std::vector<cv::Point>& Points);//将n个点的方向矫正为正上方开始，逆时针
    // 灯条四点（已正确方向排序）转换为装甲板四点
    // type:0 小装甲板，1大装甲板
    std::vector<cv::Point>light2armor(std::vector<cv::Point>& points,int type);
    //pnp解算（图像，图像点，装甲类型，先验预测中心点rtvec）
    // 其中坐标轴的原点为装甲板的中心点，x轴正方向为向右，y轴正方向为向上15度，z轴朝向摄像机向外
    // 坐标系为相机坐标系
    std::vector<RTVec> solveArmor(cv::Mat &img, std::vector<cv::Point> &points2d,int type = 0,RTVec=RTVec());
    // pnp解算（图像，图像点，装甲类型，先验预测中心点rtvec）
    // 其中坐标轴的原点为装甲板的中心点，x轴正方向为向右，y轴正方向为向上15度，z轴朝向摄像机向外
    // 使用positionTransitionChain函数，高效解算多个点。
    RTVec solveArmorPos(std::vector<cv::Point> &points2d, int type, RTVec& rtvec);
    void setCameraInfo(cv::Mat cameraMatrix,cv::Mat distCoeffs);//设置相机参数
    void setCameraInfo();//设置相机参数
    void printPoints(cv::Mat& img,std::vector<cv::Point>& correctedPoints,bool show=false,cv::Scalar color=cv::Scalar(0,255,0));// 显示点，debug用
    inline RTVec rtvec2PositionWorld(RTVec &rtvec, RTVec& cameraPositionWorld); // 转化为世界坐标系

    // 从2d点到3d全局坐标
    // type:0 小装甲板，1大装甲板
    // (图像，图像点，装甲类型,相机对世界坐标，先验预测rtvec)
    std::vector<RTVec> poins2d2Global3d(std::vector<cv::Point> &points2d, RTVec cameraPositionWorld, int type, RTVec& rtvec);
    // 世界坐标到图像坐标
    // 输入：（图像，世界点坐标组，相机相对全局方位）
    std::vector<cv::Point> global3d2points2d(cv::Mat& img, std::vector<cv::Point3f> points3d, RTVec cameraPositionWorld); 
    RTVec positionTransitionChain(std::vector<RTVec> rtvecs); // 链式坐标系变换，从全局到局部顺序
    std::vector<RTVec> positionTransitionChain(std::vector<RTVec>& rtvecs,std::vector<std::vector<float>>& basicPoints);//同时对多个点进行坐标变换；
    inline float getDistance(RTVec& target);//获取距离
    inline float getDistance(std::vector<float>& tvec);//获取距离
    inline float getDistance(Armor& armor);// 安全获取距离
    void matchArmor(cv::Mat &img, std::vector<Result> &result, RTVec cameraPositionWorld); // 匹配装甲板
    bool detectArmorsId(cv::Mat& img);
    CamControl processImage(cv::Mat& img);// 一次解决（无位置信息版本）
    

    //std::vector<cv::Point> armor2light(std::vector<cv::Point>& points);//应该用不上
private:
    InferCore core;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    inline cv::Mat rvec2rmat(std::vector<float>& rvec);//获取旋转矩阵
    RTVec getPoints3dCenter(std::vector<RTVec>& points);// 获取3d中心点
    std::vector<cv::Point3f> pos3d2Points3d(std::vector<RTVec> &pos);
    float getDistanceBbox(Armor& armor);// 单目测距大法
    void updateLiveRate(std::vector<Result>& res);// 上传存活率

    const cv::Scalar redThsd1=cv::Scalar(0,150,160);
    const cv::Scalar redThsu1 = cv::Scalar(35, 255, 255);
    const cv::Scalar redThsd2=cv::Scalar(160,100,140);
    const cv::Scalar redThsu2=cv::Scalar(180,255,255);
    const cv::Scalar blueThsd=cv::Scalar(100,100,140);
    const cv::Scalar blueThsu=cv::Scalar(124,255,255);
    // 装甲板信息，单位：m
    const float armorSWidth=0.135f;
    const float armorSHeight=0.125f;
    const float armorLWidth=0.230f;
    const float armorLHeight=0.127f;
    //其他必要信息
    const float bulletSpeed=25000.f;// 子弹速度25m/s，单位：mm/s
};

inline float getPolyIou(std::vector<cv::Point> &shape1, std::vector<cv::Point> &shape2);  // 获取两个多边形的IOU
inline float getPolyIouF(std::vector<cv::Point> &shape1, std::vector<cv::Point> &shape2); // 快速获取两个多边形的IOU（可能不准）
inline float getPolyIouF(cv::Rect &rect1, cv::Rect &rect2);                               // 矩形快速获取IOU，用于无法解算的情况

/*
祖传数据
red：
0 0 180
200 200 255
    上半圈hsv：0 100 140 ： 10 255 255
    下半圈hsv  160 100 140 ： 180 255 255

blue：
220 0 0
255 255 255
    hsv：100 100 140 ： 124 255 255


装甲板信息：
灯长55，间距135
小板宽135高125，大板宽230，高127，角度15度   comment：大板看不清灯，没法pnp解算。发现用小板的数据解算好像还挺准

*/

#endif