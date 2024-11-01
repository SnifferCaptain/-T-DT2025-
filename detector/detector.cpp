#include"detector.h"
//#include<execution>即使是c++20也用不了

Detector::Detector():core(){
    lightAreaThs=50;
    minimumIouThs=0.1f;
    judgeId=false;
    cd=3;
    realDeltaX=0.f;
    realDeltaY=0.f;
    useZero=false;
    setCameraInfo();
    accuracy();
}

Detector::Detector(std::string model_path,bool optimize,std::string NumDetectLoc):core(model_path,optimize,NumDetectLoc){
    lightAreaThs=50;
    minimumIouThs=0.1f;
    judgeId=!NumDetectLoc.empty();
    cd=3;
    realDeltaX=0.f;
    realDeltaY=0.f;
    useZero=false;
    setCameraInfo();
    accuracy();
}

void Detector::speed(){
    useFast=true;
}

void Detector::accuracy(){
    useFast=false;
}

bool Detector::loadModel(std::string model_path,bool optimize){
    return core.load(model_path,optimize);
}

cv::Mat Detector::getSubImage(cv::Mat& img,Result& result)const{
    cv::Rect rect=getRect(img,result);
    cv::Mat op;
    img(rect).copyTo(op);
    return op;
}

std::vector<cv::Point> Detector::getLightPoints(cv::Mat& img,Result& result){
    cv::Mat sub=getSubImage(img,result);
    //cv::imshow("sub",sub);
    cv::cvtColor(sub,sub,cv::COLOR_BGR2HSV);
    cv::Mat bin=cv::Mat::zeros(sub.size(),CV_8UC1);
    if(result.class_id==0){
        //蓝色
        cv::inRange(sub,blueThsd,blueThsu,bin);
    }
    else if(result.class_id==1){
        //红色
        cv::Mat orbin = cv::Mat::zeros(sub.size(), CV_8UC1);
        cv::inRange(sub,redThsd1,redThsu1,orbin);
        cv::inRange(sub,redThsd2,redThsu2,bin);
        cv::bitwise_or(orbin,bin,bin);
    }
    cv::Mat eroder;//用来判断是否为有效灯条
    bin.copyTo(eroder);
    cv::Mat thinKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
    cv::Mat binKernelE = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 4));
    cv::Mat dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 6));
    cv::erode(bin,bin,binKernelE);
    cv::dilate(bin,bin,dilateKernel);
    cv::erode(eroder,eroder,thinKernel);
    cv::dilate(eroder,eroder,dilateKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(eroder,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(auto&p : contours){
        cv::fillPoly(bin,p,cv::Scalar(255));
    }
    if(contours.size()<2){
        return std::vector<cv::Point>();
    }
    contours.clear();
    cv::findContours(bin,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    //粗筛选
    std::remove_if(contours.begin(),contours.end(),[](std::vector<cv::Point>& p){
        cv::Rect bbox=cv::boundingRect(p);
        return (bbox.width*1.7f>bbox.height);
    });
    for(int a=0;a<contours.size();a++){
        if(contours[a].size()<1){
            contours.erase(contours.begin()+a);
            a--;
        }
    }
    if (contours.size() < 2){
        return std::vector<cv::Point>();
    }
    std::sort(contours.begin(),contours.end(),[](std::vector<cv::Point>& a,std::vector<cv::Point>& b){
        return cv::contourArea(a)>cv::contourArea(b);
    });

    /*
    老模块备份
    contours.erase(contours.begin()+2,contours.end());//保留前两个
    double maxLenFixer=0.;
    std::vector<cv::Point> op;
    for(auto&p : contours){
        cv::approxPolyDP(p,p,3,true);
        double maxlength=0.;
        int maxindexA=0,maxindexB=1;
        for(int a=0;a<p.size();a++){
            for(int b=a+1;b<p.size();b++){
                double length=cv::norm(p[a]-p[b]);
                if(length>maxlength){
                    maxlength=length;
                    maxindexA=a;
                    maxindexB=b;
                }
            }
        }
        if(maxlength>maxLenFixer){
            maxLenFixer=maxlength;
        }
        op.push_back(p[maxindexA]);
        op.push_back(p[maxindexB]);
    }
    */
   

    contours.erase(contours.begin() + 2, contours.end()); // 保留前2个
    double maxLenFixer=0.;
    std::vector<cv::Point> op;//2个一组，且已经按照面积从大到小排序了
    std::vector<float> angles;// 角度,size()=op/2
    for(auto&p : contours){
        cv::approxPolyDP(p,p,3,true);
        double maxlength=0.;
        int maxindexA=0,maxindexB=1;
        for(int a=0;a<p.size();a++){
            for(int b=a+1;b<p.size();b++){
                double length=cv::norm(p[a]-p[b]);
                if(length>maxlength){
                    maxlength=length;
                    maxindexA=a;
                    maxindexB=b;
                }
            }
        }
        if(maxlength>maxLenFixer){
            maxLenFixer=maxlength;
        }
        op.push_back(p[maxindexA]);
        op.push_back(p[maxindexB]);
        angles.push_back(atan2(p[maxindexA].y-p[maxindexB].y,p[maxindexA].x-p[maxindexB].x));
    }
    // 角度匹配
    if(fabs(angles[0] - angles[1]) > CV_PI / 12.f){
        return std::vector<cv::Point>();
    }
    
    //残灯修复
    for(int i=0;i<op.size();i+=2){
        double distance = cv::norm(op[i] - op[i + 1]);
        if (distance < maxLenFixer*0.9){
            //根据中心点，将两个点之间的距离调整到maxLenFixer长度
            cv::Point midpoint = (op[i] + op[i + 1]) / 2.;
            cv::Point direction = op[i + 1] - op[i];
            cv::Point newpoint1 = midpoint + direction * maxLenFixer*0.5/distance;
            cv::Point newpoint2 = midpoint - direction * maxLenFixer*0.5/distance;
            op[i] = newpoint1;
            op[i + 1] = newpoint2;
        }
    }
    //边缘重复检测
    double lightsdistance = cv::norm(op[0] - op[3]);
    if (lightsdistance< maxLenFixer * 0.9 || lightsdistance < img.cols*0.015f){//********debug here */
        //边缘距离过近，重合
        return std::vector<cv::Point>();
    }

    //全局映射
    for(auto&p : op){
        p.x+=(result.x-result.w*0.5f)*img.cols;
        p.y+=(result.y-result.h*0.5f)*img.rows;
    }
    
    //debug
    //cv::imshow("bin", bin);
    //cv::imshow("eroder", eroder);
    // cv::waitKey(0);

    //方向矫正
    setCorrectDirection(op);
    return op;
}

std::vector<cv::Point>& Detector::setCorrectDirection(std::vector<cv::Point>& op){
    // 方案：角度or位置
    //  计算中心点
    cv::Point center(0, 0);
    for (const auto &point : op){
        center += point;
    }
    center *= (1.0 / op.size());

    // 按照角度排序
    std::sort(op.begin(), op.end(), [&](const cv::Point &a, const cv::Point &b){
        double atanA = atan2(a.y - center.y, a.x - center.x)-CV_PI/2.;
        double atanB = atan2(b.y - center.y, b.x - center.x)-CV_PI/2.;
        if(atanA < 0)atanA+=2*CV_PI;
        if(atanB < 0)atanB+=2*CV_PI;
        return atanA < atanB; 
    });
    return op;
}

inline std::vector<Result> Detector::detect(cv::Mat& img){
    return core.infer(img);
}

cv::Rect Detector::getRect(cv::Mat& img,Result& result)const{
    int rx = (result.x - result.w / 2.) * img.cols;
    int ry = (result.y - result.h / 2.) * img.rows;
    int rw = static_cast<int>(result.w * img.cols);
    int rh = static_cast<int>(result.h * img.rows);
    // 严格的安全检查
    // rx=std::max(0,rx);
    // ry=std::max(0,ry);
    // rw=std::min(rw,img.cols-rx);
    // rh=std::min(rh,img.rows-ry);
    cv::Rect rect(rx, ry, rw, rh);
    return rect;
}

std::vector<cv::Point> Detector::light2armor(std::vector<cv::Point>& points, int type){
    //  计算中心点
    cv::Point center(0, 0);
    for (const auto &point : points){
        center += point;
    }
    center *= (1.0 / points.size());
    std::vector<cv::Point> op;

    /*
    参考：
    vector<Point> Armor::fourPoints(){
        float dx1=rightTop.x-rightBottom.x+leftTop.x-leftBottom.x;
        float dy1=rightTop.y-rightBottom.y+leftTop.y-leftBottom.y;//竖向
        float dx2=rightTop.x-leftTop.x+rightBottom.x-leftBottom.x;
        float dy2=rightTop.y-leftTop.y+rightBottom.y-leftBottom.y;//横向
        //bias
        dx1*=0.55f;
        dy1*=0.55f;
        dx2*=0.20f;
        dy2*=0.20f;
        Point center;
        center.x=(rightTop.x+rightBottom.x+leftTop.x+leftBottom.x)/4;
        center.y=(rightTop.y+rightBottom.y+leftTop.y+leftBottom.y)/4;
        vector<Point> op;
        op.push_back(Point(center.x+dx1+dx2,center.y+dy1+dy2));//右上，顺时针
        op.push_back(Point(center.x-dx1+dx2,center.y-dy1+dy2));
        op.push_back(Point(center.x-dx1-dx2,center.y-dy1-dy2));
        op.push_back(Point(center.x+dx1-dx2,center.y+dy1-dy2));
        return op;
    }
    */
    float sideDx = points[0].x - points[1].x + points[3].x - points[2].x; // 左右侧的灯条x差，接近于0，为正时从上到下往左侧倾斜
    float sideDy = points[0].y - points[1].y + points[3].y - points[2].y; // 左右侧的灯条y差，一般为负
    float topDx = points[0].x - points[3].x + points[1].x - points[2].x;  // 上下侧的灯条x差，一般为负
    float topDy = points[0].y - points[3].y + points[1].y - points[2].y;  // 上下侧的灯条y差，接近于0，为正是从左到右朝上倾斜（矩形则与sideDx相反方向）
    if(type==0){
        // 小板
        constexpr float sideMul=0.25f*125.f/55.f;// 左右边的缩放系数，0.5是半边。一般掌管高度
        constexpr float topMul=0.25f*135.f/135.f;// 上下边的缩放系数，0.5是半边。一般掌管宽度
        sideDx*=sideMul;
        sideDy*=sideMul;
        topDx*=topMul;
        topDy*=topMul;
    }
    else if(type==1){
        // 大板
        constexpr float sideMul=0.25*127.f/55.f;// 左右边的缩放系数，0.5是半边。一般掌管高度
        constexpr float topMul=0.25*230.f/230.f;// 上下边的缩放系数，0.5是半边。一般掌管宽度
        sideDx*=sideMul;
        sideDy*=sideMul;
        topDx*=topMul;
        topDy*=topMul;
    }
    op.push_back(cv::Point(center.x+sideDx+topDx,center.y+sideDy+topDy));//左上角
    op.push_back(cv::Point(center.x-sideDx+topDx,center.y-sideDy+topDy));
    op.push_back(cv::Point(center.x-sideDx-topDx,center.y-sideDy-topDy));
    op.push_back(cv::Point(center.x+sideDx-topDx,center.y+sideDy-topDy));
    return op;
}

std::vector<RTVec> Detector::solveArmor(cv::Mat &img, std::vector<cv::Point> &points, int type,RTVec rtvec){
    auto& rvec=rtvec.rvec;
    auto& tvec=rtvec.tvec;
    std::vector<cv::Point3f> objectPoints(4);//装甲板中心拟合xy平面，x正方向对应向左，y正方向对应向上15度于车轴
    objectPoints[0] = cv::Point3f(-67.5f, 27.5f, 0);//左上开始，单位为毫米
    objectPoints[1] = cv::Point3f(-67.5f, -27.5f, 0);//逆时针
    objectPoints[2] = cv::Point3f(67.5f, -27.5f, 0); //***************当心用不了这个！然后记得补充type（虽然大概率我不会用上） */
    objectPoints[3] = cv::Point3f(67.5f, 27.5f, 0);
    std::vector<cv::Point2f> imagePoints(4);
    for (int i = 0; i < 4; i++){
        imagePoints[i] = points[i];
    }
    bool success=false;
    if (rvec.size() != 3 || tvec.size() != 3){
        rvec.clear();
        tvec.clear();
        success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_P3P);
    }
    else{
        success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true, cv::SOLVEPNP_P3P);
    }
    if (!success){
        std::cout << "solvePnP failed" << std::endl;
        return std::vector<RTVec>();
    }
    std::vector<RTVec> op;
    cv::Mat temptvec(3, 1, CV_32F);
    temptvec.at<float>(0, 0) = tvec[0];
    temptvec.at<float>(1, 0) = tvec[1];
    temptvec.at<float>(2, 0) = tvec[2];
    cv::Mat rotMat(3, 3, CV_32F);
    cv::Rodrigues(rvec, rotMat);
    for (int i = 0; i < 4; i++){
        cv::Mat tvecP = (cv::Mat_<float>(3, 1) << objectPoints[i].x, objectPoints[i].y, objectPoints[i].z);
        cv::Mat tvecP2 = rotMat * tvecP + temptvec;
        RTVec op0;
        op0.rvec = rvec;
        op0.tvec.push_back(tvecP2.at<float>(0, 0));
        op0.tvec.push_back(tvecP2.at<float>(1, 0));
        op0.tvec.push_back(tvecP2.at<float>(2, 0));
        op.push_back(op0);
    }
    return op;
}

RTVec Detector::solveArmorPos(std::vector<cv::Point> &points, int type, RTVec& rtvec){
    auto &rvec = rtvec.rvec;
    auto &tvec = rtvec.tvec;
    std::vector<cv::Point3f> objectPoints(4);         // 装甲板中心拟合xy平面，x正方向对应向左，y正方向对应向上15度于车轴
    objectPoints[0] = cv::Point3f(-67.5f, 65.f, 0);  // 左上开始，单位为毫米
    objectPoints[1] = cv::Point3f(-67.5f, -65.f, 0); // 逆时针
    objectPoints[2] = cv::Point3f(67.5f, -65.f, 0);//***************当心用不了这个！然后记得补充type（虽然大概率我不会用上） */
    objectPoints[3] = cv::Point3f(67.5f, 65.f, 0);
    std::vector<cv::Point2f> imagePoints(4);
    for (int i = 0; i < 4; i++){
        imagePoints[i] = points[i];
    }
    bool success = false;
    if (rvec.size() != 3 || tvec.size() != 3){
        rvec.clear();
        tvec.clear();
        success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false);//, cv::SOLVEPNP_P3P);
    }
    else{
        success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true);//, cv::SOLVEPNP_P3P);
    }
    if (!success){
        std::cout << "solvePnP failed" << std::endl;
        return RTVec();
    }
    return rtvec;
}

void Detector::setCameraInfo(){
    // default相机内参矩阵
    /*
        cameraMatrix = (cv::Mat_<float>(3, 3) <<
        1.7774091341308808e+03, 0., 7.1075979428865026e+02,
        0., 1.7754170626354828e+03, 5.3472407285624729e+02,
        0., 0., 1.
    );

    distCoeffs = (cv::Mat_<float>(5, 1) <<
        -1.0357284278115234e-01,
        1.1669449243165283e-01,
        -1.3853631656954956e-03,
        -1.2780760355858452e-04,
        -2.1767632128596191e-01
    );

    */
    cameraMatrix = (cv::Mat_<float>(3, 3) <<
        623.5383f, 0., 640.f,
        0., 1108.513f ,360.f,
        0., 0., 1.
    );
    distCoeffs = (cv::Mat_<float>(5, 1) <<
        0.f,
        0.f,
        0.f,
        0.f,
        0.f
    );
}

void Detector::setCameraInfo(cv::Mat cameraMatrix,cv::Mat distCoeffs){
    this->cameraMatrix=cameraMatrix;
    this->distCoeffs=distCoeffs;
}

void Detector::printPoints(cv::Mat& img,std::vector<cv::Point>& correctedPoints,bool show, cv::Scalar color){
    if(correctedPoints.size()==0)return;
    cv::drawContours(img, std::vector<std::vector<cv::Point>>{correctedPoints}, -1, color);
    if(show){
        cv::imshow("function:printPoints", img);
    }
}

inline RTVec Detector::rtvec2PositionWorld(RTVec& rtvec,RTVec& cameraPosition){
    std::vector<RTVec> chain;
    chain.push_back(cameraPosition);//从 全局到局部排序
    chain.push_back(rtvec);
    RTVec op = positionTransitionChain(chain);
    return op;
}

inline cv::Mat Detector::rvec2rmat(std::vector<float>& rvec){
    cv::Mat rmat;
    cv::Rodrigues(rvec, rmat);
    return rmat;
}

RTVec Detector::getPoints3dCenter(std::vector<RTVec>& rtvecs){
    RTVec op;
    op.rvec=rtvecs[0].rvec;
    op.tvec=rtvecs[0].tvec;
    for(int a=1;a<rtvecs.size();a++){
        op.rvec[0]+=rtvecs[a].rvec[0];
        op.tvec[0]+=rtvecs[a].tvec[0];
        op.rvec[1]+=rtvecs[a].rvec[1];
        op.tvec[1]+=rtvecs[a].tvec[1];
        op.rvec[2]+=rtvecs[a].rvec[2];
        op.tvec[2]+=rtvecs[a].tvec[2];
    }
    for(int a=0;a<3;a++){
        op.rvec[a]/=rtvecs.size();
        op.tvec[a]/=rtvecs.size();
    }
    return op;
}

std::vector<cv::Point3f> Detector::pos3d2Points3d(std::vector<RTVec>& pos3d){
    std::vector<cv::Point3f> op;
    for(auto& pos:pos3d){
        op.push_back(cv::Point3f(pos.tvec[0],pos.tvec[1],pos.tvec[2]));
    }
    return op;
}

RTVec Detector::positionTransitionChain(std::vector<RTVec> rtvecs){
    cv::Mat rmat=cv::Mat::eye(3,3,CV_32F);
    cv::Mat tvec=cv::Mat::zeros(3,1,CV_32F);
    // 从后往前乘
    for(int a=rtvecs.size()-1;a>=0;a--){
        auto& rtvec=rtvecs[a];
        cv::Mat r=rvec2rmat(rtvec.rvec);// to rmat
        rmat=rmat*r;// 计算rvec
        //tvec
        tvec=rmat*tvec+cv::Mat(rtvec.tvec);
    }
    RTVec op;
    cv::Rodrigues(rmat, op.rvec);// rmat转rvec
    for(int a=0;a<3;a++){
        op.tvec[a]=tvec.at<float>(a,0);
    }
    return op;
}

std::vector<RTVec> Detector::positionTransitionChain(std::vector<RTVec>& rtvecs,std::vector<std::vector<float>>& basicPoints){
    //此算法已经经过优化，无需再调整
    std::vector<RTVec> op;
    cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F);
    std::vector<cv::Mat> tvecs;// 基础点的tvec
    for(int a=0;a<basicPoints.size();a++){
        tvecs.push_back(cv::Mat(basicPoints[a]));
    }
    // 从后往前乘
    for (int a = rtvecs.size() - 1; a >= 0; a--){
        auto &rtvec = rtvecs[a];
        cv::Mat r = rvec2rmat(rtvec.rvec); // to rmat
        rmat = rmat * r;                   // 计算rvec
        // tvec
        for(int b=0;b<tvecs.size();b++){
            tvecs[b]=rmat*tvecs[b]+ cv::Mat(rtvec.tvec);
        }
    }
    std::vector<float> oprvec;
    cv::Rodrigues(rmat, oprvec); // rmat转rvec
    for(int a = 0; a < tvecs.size(); a++){
        RTVec op0;
        op0.rvec=oprvec;
        for(int b=0;b<3;b++){
            op0.tvec.push_back(tvecs[a].at<float>(b,0));
        }
        op.push_back(op0);
    }
    return op;
}

std::vector<RTVec> Detector::poins2d2Global3d(std::vector<cv::Point> &points2d, RTVec cameraPositionWorld, int type, RTVec& rtvec){
    RTVec pos2Cam = solveArmorPos(points2d, type, rtvec);
    std::vector<std::vector<float>> objectPoints;         // 装甲板中心拟合xy平面，x正方向对应向左，y正方向对应向上15度于车轴
    {
        std::vector<float> basicPoint1({-67.5f, 27.5f, 0}); // 左上开始，单位为毫米
        std::vector<float> basicPoint2({-67.5f, -27.5f, 0});//逆时针
        std::vector<float> basicPoint3({67.5f, -27.5f, 0});
        std::vector<float> basicPoint4({67.5f, 27.5f, 0});
        objectPoints.push_back(basicPoint1);
        objectPoints.push_back(basicPoint2);
        objectPoints.push_back(basicPoint3);
        objectPoints.push_back(basicPoint4);
    } 
    
    std::vector<RTVec> chain;
    chain.push_back(cameraPositionWorld);
    chain.push_back(pos2Cam);
    std::vector<RTVec> posGlobal=positionTransitionChain(chain,objectPoints);
    return posGlobal;
}

inline float Detector::getDistance(RTVec& target){
    return getDistance(target.tvec);
}

inline float Detector::getDistance(std::vector<float>& tvec){
    if(tvec.size()<3)return -1;
    float distance = sqrtf(powf(tvec[0], 2) + powf(tvec[1], 2) + powf(tvec[2], 2));
    return distance;
}

inline float Detector::getDistance(Armor& armor){
    if(armor.lastCenter.tvec.size()>3)return getDistance(armor.lastCenter.tvec);
    if(armor.point2d.size()<4)return getDistanceBbox(armor);
    solveArmorPos(armor.point2d,0,armor.lastCenter);
    float dist=getDistance(armor.lastCenter);
    std::cout<< "[distance]: "<< dist << " [bbox]: w:"<< armor.bbox.width<<" h:"<< armor.bbox.height << " area:"<< armor.bbox.area() << std::endl;
    return getDistance(armor.lastCenter);
}

float Detector::getDistanceBbox(Armor& armor){
    if(armor.bbox.empty())return -1;
    float height=armor.bbox.height;// 横向可以骗我，但是纵向可不会
    float fy=cameraMatrix.at<float>(1, 1);// height方向得用y方向的焦距
    // height*distance=fixed
    float xdy=armor.bbox.width/armor.bbox.height;//1.44 75000 std
    float fixed=60000*(4.f/(xdy+3.f));
    float distance=fixed/height;
    return distance;
}

void Detector::updateLiveRate(std::vector<Result>& res){
    if(res.size()==0){
        // 当做全部存活
        liveRate=liveRate*0.8f+0.2f;
        return;
    }
    float death=0.f;
    for(auto& result: res){
        if(result.class_id==2){
            death++;
        }
    }
    death=1.f-death/static_cast<float>(res.size());//成存活率了
    liveRate=liveRate*0.85f+0.15f*death;
}

inline float getPolyIou(std::vector<cv::Point>& first, std::vector<cv::Point>& second){
    float contourAreaFirst=cv::contourArea(first);
    float contourAreaSecond=cv::contourArea(second);
    std::vector<cv::Point> intersection;
    float inter=cv::intersectConvexConvex(first, second, intersection);
    return inter/(contourAreaFirst+contourAreaSecond-inter);
}

inline float getPolyIouF(std::vector<cv::Point>& first, std::vector<cv::Point>& second){
    //获取最小外接矩形
    cv::Rect rectFirst= cv::boundingRect(first);
    cv::Rect rectSecond= cv::boundingRect(second);
    float area1=rectFirst.area();
    float area2=rectSecond.area();
    float area=(rectFirst&rectSecond).area();
    return area/(area1+area2-area);
}

inline float getPolyIouF(cv::Rect& r1,cv::Rect& r2){
    float area1=r1.area();
    float area2=r2.area();
    float area=(r1&r2).area();
    return area/(area1+area2-area);
}

std::vector<cv::Point> Detector::global3d2points2d(cv::Mat& img, std::vector<cv::Point3f> points3d, RTVec cameraPositionWorld){
    std::vector<cv::Point> points2d;
    cv::projectPoints(points3d,cameraPositionWorld.rvec,cameraPositionWorld.tvec,cameraMatrix,distCoeffs,points2d);
    return points2d;
}

void Detector::matchArmor(cv::Mat &img, std::vector<Result> &results, RTVec cameraPositionWorld){
    //NOTE:匹配逻辑：
    // 1、将detect分类，能getLightPoints成功的优先（因为能够解算iou）。分为A：可解算，B：不可解算
    // 2、将A可解算拿出来，解算为装甲板点组
    // 3、首先将老装甲板卡尔曼预测
    // 4、使用getPolyIou进行匹配
    //    1、从老匹配A组装甲板，iou大于阈值，且选择iou最大的进行匹配 #无姿态装甲板使用getPolyIouF匹配#
    //    2、没人要的老装甲板，放入B组等待匹配
    //    3、没人要的A组装甲板，创建新装甲板
    // 5、使用getPolyIouF进行匹配
    //    1、从老匹配B组装甲板，iou大于阈值，且选择iou最大的进行匹配
    //    2、没人要的老装甲板，进入衰退期（置信度下降）
    //    3、没人要的B组装甲板，创建新装甲板（不包含点组信息，因此置信度要打折）
    // 6、match后的装甲板进行更新。
    // 0、数字预测不在这里

    //分组&解算
    std::vector<ArmorTemp> groupA,groupB;
    for(auto& result:results){
        std::vector<cv::Point> lights=getLightPoints(img,result);
        if(lights.size()==0){
            // group B
            ArmorTemp item;
            item.bbox=result.toCvRect(img.cols,img.rows);
            item.haveFacing=false;
            item.havePose3d=false;
            item.score=result.conf;
            //item.color=result.class_id;
            item.addScoreViaShape(result.w,result.h);
            groupB.push_back(item);
        }
        else{
            // group A
            std::vector<cv::Point> armorPoints = light2armor(lights, 0);
            ArmorTemp item;
            item.bbox = result.toCvRect(img.cols, img.rows);
            item.haveFacing = true;
            item.point2d = armorPoints;
            item.score = result.conf;
            if (result.class_id == 2){
                // 死了
                item.score *= 0.3;
            }
            // item.color=result.class_id;
            item.addScoreViaLight();
            item.addScoreViaShape(result.w,result.h);
            groupA.push_back(item);
        }

    }
    // 2、老装甲板预测 并 匹配A组 & B组    ####三维模式有内存问题####

    std::vector<float> ctrl;// 控制向量
    ctrl.push_back(realDeltaX);
    ctrl.push_back(realDeltaY);

    int unmatch=0;// 不匹配数
    for(int a=0;a<armors.size();a++){
        ArmorTemp armor;
        if(useFast){
            armor = armors[a].predictNext(cameraMatrix, distCoeffs, ctrl);
        }
        else{
            armor=armors[a].predictNext(cameraMatrix, distCoeffs, cameraPositionWorld);
        }
        if(armor.haveFacing){
            // iou匹配
            int index=0;
            float iouMax=0.f;
            for(int b=0;b<groupA.size();b++){
                float iou=getPolyIouF(armor.bbox,groupA[b].bbox);//改为纯bbox匹配方案
                if(iou>iouMax){
                    index=b;
                    iouMax=iou;
                }
            }
            if(iouMax>minimumIouThs){
                // 匹配A成功  Target为最新匹配点，armors[a]为维护列表
                // 解算3d
                ArmorTemp& target = groupA[index];
                if (useFast){
                    target.havePose3d = false;
                }
                else{
                    target.havePose3d = true;
                    std::vector<RTVec> pos3d=poins2d2Global3d(target.point2d,cameraPositionWorld,0,armors[a].lastCenter);// rtvec初始值能够快速迭代，不是变得精确
                    target.pos3d=pos3d2Points3d(pos3d);
                }
                armors[a].updateNew(target,ctrl);
                float dist=getDistance(armors[a]);
                armors[a].updateDistance(dist,0.01f);
                groupA.erase(groupA.begin()+index);
            }
            else{
                // 匹配A失败 匹配B组
                index=0;
                iouMax=0.f;
                for(int b=0;b<groupB.size();b++){
                    float iou=getPolyIouF(armor.bbox,groupB[b].bbox);
                    if(iou>iouMax){
                        index=b;
                        iouMax=iou;
                    }
                }
                if(iouMax>minimumIouThs){
                    // 匹配B成功
                    ArmorTemp& target = groupB[index];
                    armors[a].updateNew(target,ctrl);
                    float dist = getDistance(armors[a]);// 可能面临无4p的风险
                    armors[a].updateDistance(dist,0.05f);
                    groupB.erase(groupB.begin()+index);
                }
                else{
                   // 匹配B失败，得分减少&消除
                    if(armors[a].score<0.3){
                        armors.erase(armors.begin()+a);
                        a--;
                        continue;
                    }
                    armors[a].lost();
                    unmatch++;
                }
            }
        }
        else{
            // 当前老armor没pos
            int index=0;
            float iouMax=0.f;
            for(int b=0; b<groupA.size();b++){
                float iou=getPolyIouF(armor.bbox,groupA[b].bbox);
                if(iou>iouMax){
                    index=b;
                    iouMax=iou;
                }
            }
            if(iouMax>minimumIouThs){
                // 匹配A成功
                ArmorTemp& target = groupA[index];
                if(useFast){
                    target.havePose3d = false;
                }
                else{
                    target.havePose3d = true;
                    RTVec& posCam=armors[a].lastCenter;
                    std::vector<RTVec> pos3d = poins2d2Global3d(armor.point2d, cameraPositionWorld, 0,posCam);
                    target.pos3d = pos3d2Points3d(pos3d);
                }
                armors[a].updateNew(target,ctrl);
                float dist = getDistance(armors[a]);// 此时有pos
                armors[a].updateDistance(dist,0.01f);
                groupA.erase(groupA.begin()+index);
            }
            else{
                // 匹配A失败，匹配B
                index=0;
                iouMax=0.f;
                for(int b=0;b<groupB.size();b++){
                    float iou=getPolyIouF(armor.bbox,groupB[b].bbox);
                    if(iou>iouMax){
                        index=b;
                        iouMax=iou;
                    }
                }
                if(iouMax>minimumIouThs){
                    // 匹配B成功
                    ArmorTemp& target = groupB[index];
                    armors[a].updateNew(target,ctrl);
                    float dist = getDistance(armors[a]);// 此时一定没有p4
                    armors[a].updateDistance(dist,0.05f);
                    groupB.erase(groupB.begin()+index);
                }
                else{
                   // 匹配B失败，得分减少&消除
                    if(armors[a].score<0.3){
                        armors.erase(armors.begin()+a);
                        a--;
                        continue;
                    }
                    armors[a].lost();
                    unmatch++;
                }
            }
        }
        
    }
    // 零匹配
    if(unmatch>=armors.size() && useZero && unmatch!=0){
        // A组一定在前
        for(int a=0;a<armors.size();a++){
            if(groupA.size()>0){
                ArmorTemp &target = groupA[0];
                if (useFast)
                {
                    target.havePose3d = false;
                }
                else
                {
                    target.havePose3d = true;
                    std::vector<RTVec> pos3d = poins2d2Global3d(target.point2d, cameraPositionWorld, 0, armors[a].lastCenter); // rtvec初始值能够快速迭代，不是变得精确
                    target.pos3d = pos3d2Points3d(pos3d);
                }
                armors[a].updateNew(target, ctrl);
                float dist = getDistance(armors[a]);
                armors[a].updateDistance(dist, 0.003f);
                groupA.erase(groupA.begin());
            }
            else if(groupB.size()>0){
                ArmorTemp &target = groupB[0];
                armors[a].updateNew(target, ctrl);
                float dist = getDistance(armors[a]);
                armors[a].updateDistance(dist, 0.01f);
                groupB.erase(groupB.begin());
            }
            else{
                break;
            }
        }
        //debug
        std::cout<<"Matched Not!"<<std::endl;
    }
    else{
        std::cout<<"Matched!"<<std::endl;
    }
    // 3、剩余A组创建新装甲板
    for(int a=0;a<groupA.size();a++){
        // A组需要先补全3d
        ArmorTemp &target = groupA[a];
        RTVec posCam;
        if (useFast){
            target.havePose3d = false;
        }
        else{
            target.havePose3d = true;
            std::vector<RTVec> pos3d = poins2d2Global3d(target.point2d, cameraPositionWorld, 0, posCam); // rtvec初始值能够快速迭代，不是变得精确
            target.pos3d=pos3d2Points3d(pos3d);
        }
        armors.push_back(Armor(target,useFast));
        armors.back().lastCenter=posCam;
        float dist = getDistance(armors.back());// 此时一定有pos
        armors.back().updateDistance(dist,0.01f);
    }
    // 4、剩余B组创建新装甲板
    for(int a=0;a<groupB.size();a++){
        ArmorTemp &target = groupB[a];
        armors.push_back(Armor(target,useFast));
        float dist = getDistance(armors.back());// 此时一定没有p4
        armors.back().updateDistance(dist,0.01f);
        //rtvec啥也没有其实也不影响
    }
    // kf后更新2d点坐标
    for(auto& armor:armors){
        if(armor.facing>0 && armor.useFast && armor.pos3dGlobal.size()>0){
            // 表示已经启用3d解算
            armor.point2d=global3d2points2d(img,armor.pos3dGlobal,cameraPositionWorld);
        }
    }
    //更新id
    if(judgeId){
        detectArmorsId(img);
    }
    //end
}

bool Detector::detectArmorsId(cv::Mat& img){
    if(!judgeId)return false;
    if(!core.idAvaliable)return false;
    for(auto& armor:armors){
        cv::Rect roi=armor.bbox;
        roi.x+=roi.width*0.2;
        roi.width*=0.6;
        cv::Mat subNum;
        try{
            img(roi).copyTo(subNum);
            armor.updateId(core.inferId(subNum));
            //cv::imshow("subNum",subNum);
        }catch(const cv::Exception& e){}

    }
    return true;
}

CamControl Detector::processImage(cv::Mat& img){
    // ###不能调整图片宽高，否则pnp会变###
    CamControl op;
    std::vector<Result> detresult = detect(img);
    updateLiveRate(detresult);
    //debug 
    //core.postResult(img,detresult);
    std::vector<std::vector<cv::Point>> lights;
    RTVec cameraPosFake;
    {
        cameraPosFake.rvec.push_back(0);
        cameraPosFake.rvec.push_back(0);
        cameraPosFake.rvec.push_back(0);
        cameraPosFake.tvec.push_back(0);
        cameraPosFake.tvec.push_back(0);
        cameraPosFake.tvec.push_back(0);
    }
    matchArmor(img, detresult, cameraPosFake);
    if(armors.size()==0){
        // 没有检测到装甲板,大陀螺启动！

        op.deltaPitch=0;
        op.deltaYaw=15*cd;
        op.haveTarget=false;
        if(liveRate>0.25f)
            op.dead=false;
        else op.dead=true;
        if(cd<3){
            cd++;
        }
        return op;
    }
    int maxIndex=0;
    for (int a=0;a<armors.size();a++){
        if(armors[a].score>armors[maxIndex].score && armors[a].distance<4000){
            maxIndex=a;
        }
    }
    cv::Point2f targetPoint(-1,-1);
    float distmm = armors[maxIndex].distance;
    //cd=0;// cd机制已经过时了
    if(cd<=0){
        float time = distmm / bulletSpeed; // 单位：s  ##帧率不够的话还需要帧率补偿##
        time+=0.067f+0.05f;// 帧率补偿+延迟补偿
        if(useFast){
            // 相机运动方程
            std::cout<<"using ctrlVec : [dx]"<< realDeltaX<<"  [dy]:" << realDeltaY<<std::endl;
            std::vector<float> ctrlVec;
            ctrlVec.push_back(realDeltaX*1.f);
            ctrlVec.push_back(realDeltaY*1.f);
            targetPoint = armors[maxIndex].predictTarget(time, cameraMatrix, distCoeffs, ctrlVec);
        }
        else{
            targetPoint = armors[maxIndex].predictTarget(time, cameraMatrix, distCoeffs, cameraPosFake);
        }
        armors[maxIndex].addScoreViaShoot();//未知原因导致问题
    }
    else{
        targetPoint.x=armors[maxIndex].bbox.x+armors[maxIndex].bbox.width/2;
        targetPoint.y=armors[maxIndex].bbox.y+armors[maxIndex].bbox.height/2;
        if(cd>3){
            //armors.clear();
        }

        cd--;
    }
    
    if(targetPoint.x==-1 || targetPoint.y==-1){
        std::cout<<"检测错误"<<std::endl;
        op.deltaPitch=0;
        op.deltaYaw=20;
        op.haveTarget=false;
        if(liveRate>0.20f)
            op.dead=false;
        else op.dead=true;
        return op;
    }
    float deltax = img.cols / 2 - targetPoint.x + realDeltaX * 0.2f;
    float deltay = img.rows / 2 - targetPoint.y + realDeltaY * 0.2f;
    std::cout<<"targetPoint:"<<targetPoint<<std::endl;
    std::cout<<"distance mm:"<<distmm<<std::endl;
    // NOTE: delta=tan(deltaAng)*focus   ==>  deltaAng=atan(delta/focus)
    op.deltaYaw = -std::atan(deltax / cameraMatrix.at<float>(0, 0)) * 180 / CV_PI;
    op.deltaPitch = std::atan(deltay / cameraMatrix.at<float>(1, 1)) * 180 / CV_PI;

    // 零匹配补偿
    //if(abs(op.deltaYaw) > 15){
    //    op.deltaYaw *= 1.15f;
    //}
    //if(abs(op.deltaYaw) > 5){
    //    op.deltaYaw *= 1.05f;
    //}

    op.haveTarget=true;
    if(liveRate>0.25f)
        op.dead=false;
    else op.dead=true;
    // 抛物线补偿
    op.deltaPitch+=distmm/3500.f;
    
    // 跟随退后
    if(distmm>2200){
        op.deltaForward=1.5f;
    }
    else if(distmm>1600){
        op.deltaForward=0.7f;
    }
    else if(distmm<800){
        op.deltaForward=-3.f;
    }
    else if(distmm<1300){
        op.deltaForward=-1.f;// 退后
    }
    else if(distmm<1300&&realDeltaY<-5){
        op.deltaForward=-1.2f;// 出生突脸
    }
    else{
        op.deltaForward = 0;
    }
    
    // 过度移动补偿
    if(abs(op.deltaPitch)>20)op.deltaPitch=20*op.deltaPitch/abs(op.deltaPitch);
    if(abs(op.deltaYaw)>60)op.deltaYaw=60*op.deltaYaw/abs(op.deltaYaw);
    realDeltaX=std::tan(op.deltaYaw*CV_PI/180.f)*cameraMatrix.at<float>(0, 0);
    realDeltaY=std::tan(op.deltaPitch*CV_PI/180.f)*cameraMatrix.at<float>(1, 1);
    if(op.deltaYaw>15){
        op.deltaH=3.f;// 节约子弹不在这里！
    }
    else{
        op.deltaH=0;
    }

    //debug
    //cv::circle(img, targetPoint, 8, cv::Scalar(0, 0, 255), -1);
    return op;

}


void alt0(){
    //测试
    Detector detector("../armor3-1300.onnx", false);
    detector.speed();
    cv::VideoCapture cap("/home/mrye/Desktop/TDT/T-DT2023-OpenCV/task1/build/Infantry_red.avi");
    cv::Mat frame;
    while(cap.read(frame)){
        std::vector<Result> detresult= detector.detect(frame);
        std::vector<std::vector<cv::Point>> lights;
        for(auto& result:detresult){
            std::vector<cv::Point> light(detector.getLightPoints(frame,result));
            if (light.size() != 0){
                detector.printPoints(frame, light, false);
                std::vector<cv::Point> armorPoints = detector.light2armor(light, 0);
                detector.printPoints(frame, armorPoints, false);
                //RTVec armorPos = detector.solveArmor(frame, armorPoints);
                //float armorDistance = detector.getDistance(armorPos);
            }
            result.draw(frame);
        }

        cv::imshow("frame", frame);
        cv::waitKey(1);
    }
}

void alt1(){
    //测试
    //Detector detector("../armor3-1300.onnx", false,"../special.onnx");
    Detector detector("/home/mrye/Desktop/School War/models/f32.onnx", true);
    detector.speed();// acc先不修了，speed能用效果好
    //cv::VideoCapture cap("/home/mrye/Desktop/TDT/T-DT2023-OpenCV/task1/build/Infantry_red.avi");
    cv::VideoCapture cap("/home/mrye/Videos/Screencasts/detectTry.webm");
    cv::Mat frame;
    while(cap.read(frame)){
        //###不能调整图片宽高，否则pnp会变###
        std::vector<Result> detresult= detector.detect(frame);
        std::vector<std::vector<cv::Point>> lights;
        RTVec cameraPosFake;
        {
            cameraPosFake.rvec.push_back(0);
            cameraPosFake.rvec.push_back(0);
            cameraPosFake.rvec.push_back(0);
            cameraPosFake.tvec.push_back(0);
            cameraPosFake.tvec.push_back(0);
            cameraPosFake.tvec.push_back(0);
        }
        detector.matchArmor(frame,detresult,cameraPosFake);
        for(auto& arm: detector.armors){
            //armorpos
            if(arm.facing>=0.5){
                detector.printPoints(frame, arm.point2d, false);
                //float distmm=detector.getDistance(arm);
                float distmm=arm.distance;
                cv::putText(frame,std::to_string(distmm),arm.bbox.tl(),cv::FONT_HERSHEY_SIMPLEX,1.5,cv::Scalar(0,255,0),1);
            }
                
            arm.printBbox(frame);
            //##### 重要！只有facing>0.5才进行计算 ###
        }
        for(auto& result:detresult){
            //bbox
            //result.draw(frame);
        }
        
        cv::imshow("frame", frame);
        cv::waitKey(0);
    }
}

int Dmain(){
    alt1();
}
