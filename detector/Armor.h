#pragma once
#include<opencv2/opencv.hpp>
#include<vector>
#include"../carman/carman.hpp"

/*
RULE:
计分规则：
基础分：置信度。
成功灯条匹配（成功姿态解算）：x1.4
成为打击目标：x1.3
继承decay：加权平均数0.8老0.2新。（或许可以改成0.9）
*/

#ifndef ARMOR_H
#define ARMOR_H


struct RTVec{
    std::vector<float> rvec;
    std::vector<float> tvec;
    RTVec(){}
};

// 临时装甲板，update到Armor列表中
class ArmorTemp{
public:    
    bool haveFacing=false;//能够进行姿态解算
    bool havePose3d=false;//进行3d姿态解算
    std::vector<cv::Point> point2d;// 姿态点组（4x）
    std::vector<cv::Point3f> pos3d;// 3d姿态组（4x）
    cv::Rect bbox;// 基础矩形框
    float score=0.f;//预测的置信度 & 通过灯条匹配加分
    int id;// 装甲板数字
    //int color;// 装甲板颜色，来源于Result

    void addScoreViaLight(); // 通过灯条匹配加分
    void addScoreViaShape(float w, float h); // 通过朝向加分
};


class Armor{
/*
NOTE：
1、只有姿态装甲板才有3d卡尔曼滤波。
2、同时存在2d与3d卡尔曼滤波，选择3d卡尔曼滤波结果，并使用3d卡尔曼结果更新2d（需求：facing>1）。
*/
    
public:
    int id;// 装甲板数字
    cv::Rect bbox;// 基本矩形框
    float score;// 得分
    float facing;// 姿态解算容忍度，多帧无姿态后，退化为无姿态装甲板，并降低得分
    bool useFast;// 是否使用快速模式，快速模式下，只使用2d卡尔曼，不进行3d卡尔曼预测
    std::vector<Carman> carmans; // 2d（xywh，p=2）+2dpos(4*point2d(xy,p=2)) or 3dpos（4*Point3d（xyz，p=2））
    float distance;// 预测距离

    
    RTVec lastCenter;// 上一帧中心点，相机坐标系
    std::vector<cv::Point> point2d;// 屏幕姿态点组（4x）
    std::vector<cv::Point3f> pos3dGlobal;// 全局姿态组（4x）属于私有维护变量，使用point2d进行调用。
    
    Armor();
    Armor(ArmorTemp& create,bool useSpeedMode);
    ArmorTemp predictNext(cv::Mat cameraMatrix, cv::Mat distCoeffs, RTVec cameraPosition = RTVec()); // 预测下一帧，用于update，假定帧率相同，不传内参就是2d预测
    ArmorTemp predictNext(cv::Mat cameraMatrix, cv::Mat distCoeffs, std::vector<float> control); // 预测下一帧，用于update，仅2d
    cv::Point2f predictTarget(float time,cv::Mat cameraMatrix, cv::Mat distCoeffs,RTVec cameraPosition=RTVec());// 预测目标点（传入时间需要包含延迟！） ， -1，-1为没有目标
    cv::Point2f predictTarget(float time, cv::Mat cameraMatrix, cv::Mat distCoeffs, std::vector<float> control); // 带相机运动预测(2d专用)
    void updateNew(ArmorTemp& recent,std::vector<float> control);// 成功匹配后更新
    void updateNew(ArmorTemp& recent);// old & wasted(3d可能用得上)
    void updateDistance(float _distance,float noise);// 更新距离 & 分数 单目测距需要调低信任
    void lost();// 丢失
    void updateId(int id);// 更新id
    void printBbox(cv::Mat& img);

    void addScoreViaShoot(); // 通过连续击打加分
    void addScoreViaDistance();// 通过距离加分
    // 应该还有一个旋转加分的，但是由于摄像机原因暂时不考虑
private:
    //  ### 转换函数均没有安全检查 ###
    std::vector<float> predictNext(float time,cv::Mat cameraMatrix, cv::Mat distCoeffs, RTVec cameraPosition);
    std::vector<float> bbox2CarmanData()const;// 将bbox转化为Carman数据
    std::vector<float> bbox2CarmanData(cv::Rect bbox) const;            // 将bbox转化为Carman数据
    cv::Rect carmanData2Bbox(std::vector<float> data)const;// 将Carman数据转化为bbox 使用xy center！！
    std::vector<float> point3d2CarmanData(cv::Point3f p)const;// 将point3d转化为Carman数据
    cv::Point3f carmanData2Point3d(std::vector<float> data)const;// 将Carman数据转化为point
    cv::Point carmanData2Point2d(std::vector<float> data)const;// 将Carman数据转化为point2d
    std::vector<float> point2d2CarmanData(cv::Point p)const;// 将point2d转化为Carman数据
    RTVec carmanData2Rtvec(std::vector<float> data) const;// 将carman数据转化为RTVec
    std::vector<float> rtvec2CarmanData(RTVec rtvec)const;// 将RTVec转化成carman数据
    unsigned int ids[10];// id数量存储
    int facingMaximum=2;// 最大允许的连续帧数
    float videoFGTime=0.07f;// 30fps+，初始化时修改
    float delayTime = 0.05f; // 帧延迟0.1
    Carman distCarman;// 测距卡尔曼滤波器
};

#endif