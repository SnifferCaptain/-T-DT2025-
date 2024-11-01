#include"Armor.h"

void ArmorTemp::addScoreViaLight(){
    score *= 1.3f;    
}

void ArmorTemp::addScoreViaShape(float w,float h){
    float rate=0.4f/(1.f-std::min(w,h)/std::max(w,h));// 前项数字越小，缩放倍率越大
    score*=1.f+rate;
}

void Armor::addScoreViaShoot(){
    score = score + 0.2f;
}

Armor::Armor():ids({}),distCarman(videoFGTime,3,1,0,1.f){
    distance=-1;
}

Armor::Armor(ArmorTemp& create,bool speedMode):ids({}),distCarman(videoFGTime,3,1,0,0.05f){
    // creating a new obj
    id = create.id;
    bbox=create.bbox;
    score=create.score;
    facing=facingMaximum;
    useFast = speedMode;
    distance=-1;
    std::vector<float> bboxData=bbox2CarmanData();
    carmans.emplace_back(videoFGTime,2,4,2,0.005f);//bboxKF
    carmans.back().initialize(bboxData);
    if(create.haveFacing){
        point2d=create.point2d;
        pos3dGlobal=create.pos3d;
        if(useFast || !create.havePose3d){
            //kf二维点
            for(int a=0;a<4;a++){
                carmans.emplace_back(videoFGTime,2,2,2,0.005f);
                std::vector<float> pointData=point2d2CarmanData(point2d[a]);
                carmans.back().initialize(pointData);
            }
        }
        else{
            //三维点组
            for(int a=0;a<4;a++){
                carmans.emplace_back(videoFGTime,2,3);
                std::vector<float> pointData=point3d2CarmanData(pos3dGlobal[a]);
                carmans.back().initialize(pointData);
            }
        }
    }
    else{
        facing=0;
    }
}

ArmorTemp Armor::predictNext(cv::Mat cameraMatrix, cv::Mat distCoeffs, RTVec cameraPosition){
    if(carmans.size()==0){
        //神奇错误，但是可以避免闪退
        ArmorTemp temp;
        temp.bbox=bbox;
        if(facing>0){
            temp.haveFacing = true;
            temp.point2d = point2d;
            return temp;
        }
        temp.haveFacing = false;
        temp.point2d.clear();
        return temp;
    }
    ArmorTemp temp;
    temp.score=0.f;
    temp.bbox = carmanData2Bbox(carmans[0].predict(videoFGTime+delayTime));
    if (facing <= 0.3 || cameraPosition.tvec.empty()){
        // 无姿态或者姿态容忍为1（也就是即将销毁）或者没有摄像头信息
        temp.haveFacing=false;
        temp.havePose3d=false;
        return temp;
    }
    temp.haveFacing=true;
    if(useFast){
        // 2dkf
        temp.havePose3d=false;
        temp.point2d.clear();
        for (int a = 1; a < carmans.size(); a++){
            cv::Point p = carmanData2Point2d(carmans[a].predict(videoFGTime+delayTime));
            temp.point2d.push_back(p);
        }
    }
    else{
        // 3dkf
        temp.havePose3d=true;
        std::vector<cv::Point3f> points3d;
        for (int a = 1; a < carmans.size(); a++){
            points3d.push_back(carmanData2Point3d(carmans[a].predict(videoFGTime+delayTime)));
            
        }
        cv::Mat tmpPoint2d;
        // 3d转2d
        cv::projectPoints(points3d, cameraPosition.rvec, cameraPosition.tvec, cameraMatrix, distCoeffs, tmpPoint2d);
        for(int a=0;a<tmpPoint2d.cols;a++){
            temp.point2d.push_back(cv::Point(tmpPoint2d.at<float>(a,0),tmpPoint2d.at<float>(a,1)));
        }
    }

    return temp;
}

ArmorTemp Armor::predictNext(cv::Mat cameraMatrix, cv::Mat distCoeffs, std::vector<float> control){
    if(carmans.size()==0){
        //神奇错误，但是可以避免闪退
        ArmorTemp temp;
        temp.bbox=bbox;
        if(facing>0){
            temp.haveFacing = true;
            temp.point2d = point2d;
            return temp;
        }
        temp.haveFacing = false;
        temp.point2d.clear();
        return temp;
    }
    ArmorTemp temp;
    temp.score=0.f;
    temp.bbox = carmanData2Bbox(carmans[0].predict(videoFGTime+delayTime,control));
    if (facing <= 0.3 || !useFast){
        // 无姿态或者用错了函数（也就是即将销毁）或者没有摄像头信息
        temp.haveFacing=false;
        temp.havePose3d=false;
        return temp;
    }
    temp.haveFacing=true;
    // 2dkf
    temp.havePose3d = false;
    temp.point2d.clear();
    for (int a = 1; a < carmans.size(); a++){
        cv::Point p = carmanData2Point2d(carmans[a].predict(videoFGTime+delayTime, control));
        temp.point2d.push_back(p);
    }
    return temp;
}

cv::Point2f Armor::predictTarget(float time,cv::Mat cameraMatrix, cv::Mat distCoeffs,std::vector<float> control){
    cv::Point2f op(-1,-1);
    if (facing <= 0.5 || carmans.size()<5){
        // 无姿态，使用bbox大法
        if(carmans.size()<1)return op;
        cv::Rect bboxFuture=carmanData2Bbox(carmans[0].predict(time,control));
        op.x=bboxFuture.x+bboxFuture.width/2;
        op.y=bboxFuture.y+bboxFuture.height/2;
        return op;
    }
    else if(useFast){
        // 2d
        std::vector<cv::Point> point2dsFuture;
        for(int a=1;a<carmans.size();a++){
            point2dsFuture.push_back(carmanData2Point2d(carmans[a].predict(time,control)));
        }
        op.x=point2dsFuture[0].x+point2dsFuture[1].x+point2dsFuture[2].x+point2dsFuture[3].x;
        op.y=point2dsFuture[0].y+point2dsFuture[1].y+point2dsFuture[2].y+point2dsFuture[3].y;
        op/=4;
        return op;
    }
    else{
        return op;
    }
}

cv::Point2f Armor::predictTarget(float time,cv::Mat cameraMatrix, cv::Mat distCoeffs,RTVec cameraPosition){
    cv::Point2f op(-1,-1);
    if (facing <= 0.5 || carmans.size()<5){
        // 无姿态，使用bbox大法
        if(carmans.size()<1)return op;
        cv::Rect bboxFuture=carmanData2Bbox(carmans[0].predict(time));
        op.x=bboxFuture.x+bboxFuture.width/2;
        op.y=bboxFuture.y+bboxFuture.height/2;
        return op;
    }
    else if(useFast || cameraPosition.tvec.empty()){
        // 2d
        std::vector<cv::Point> point2dsFuture;
        for(int a=1;a<carmans.size();a++){
            point2dsFuture.push_back(carmanData2Point2d(carmans[a].predict(time)));
        }
        op.x=point2dsFuture[0].x+point2dsFuture[1].x+point2dsFuture[2].x+point2dsFuture[3].x;
        op.y=point2dsFuture[0].y+point2dsFuture[1].y+point2dsFuture[2].y+point2dsFuture[3].y;
        op/=4;
        return op;
    }
    else{
        // 3d
        std::vector<cv::Point3f> points3dFuture;
        for(int a=1;a<carmans.size();a++){
            points3dFuture.push_back(carmanData2Point3d(carmans[a].predict(time)));
        }
        cv::Mat tmpPoint2d;
        cv::projectPoints(points3dFuture, cameraPosition.rvec, cameraPosition.tvec, cameraMatrix, distCoeffs, tmpPoint2d);
        op.x=tmpPoint2d.at<float>(0,0)+tmpPoint2d.at<float>(1,0)+tmpPoint2d.at<float>(2,0)+tmpPoint2d.at<float>(3,0);
        op.y=tmpPoint2d.at<float>(0,1)+tmpPoint2d.at<float>(1,1)+tmpPoint2d.at<float>(2,1)+tmpPoint2d.at<float>(3,1);
        op/=4;
        return op;
    }
}

void Armor::updateNew(ArmorTemp& other){
    //score update
    score=other.score*0.2f+score*0.8f;
    std::vector<float> bboxData0=bbox2CarmanData(other.bbox);
    // kf更新bbox
    std::vector<float> bboxCorrected=carmans[0].correct(bboxData0,other.score);
    bbox=carmanData2Bbox(bboxCorrected);
    if(other.haveFacing){
        if(facing<=0){
            // create new (no safty check)
            if(useFast || !other.havePose3d){
                // create 2d
                for(int a=0;a<4;a++){
                    carmans.emplace_back(videoFGTime);
                    std::vector<float> pointData=point2d2CarmanData(other.point2d[a]);
                    carmans.back().initialize(pointData);
                }
                
            }
            else{
                for (int a = 0; a < 4; a++){
                    carmans.emplace_back(videoFGTime,2,3);
                    std::vector<float> pointData = point3d2CarmanData(other.pos3d[a]);
                    carmans.back().initialize(pointData);
                }
            }
        }
        else{
            // update
            if (useFast || !other.havePose3d){
                point2d.clear();
                for (int a = 1; a < carmans.size(); a++){
                    std::vector<float> pointData=point2d2CarmanData(other.point2d[a-1]);
                    std::vector<float> correctedData = carmans[a].correct(pointData);
                    point2d.push_back(carmanData2Point2d(correctedData));
                }
            }
            else{
                for (int a = 1; a < carmans.size(); a++){
                    std::vector<float> pointData = point3d2CarmanData(other.pos3d[a-1]);
                    std::vector<float> correctedData = carmans[a].correct(pointData);
                    pos3dGlobal[a-1]=carmanData2Point3d(correctedData);
                    //解算2d点不在这里
                }
            }
        }
        facing+=facing<facingMaximum?0.3:0;
    }
    else{
        facing--;
        if(facing<0){
            facing=0;
            point2d.clear();
            pos3dGlobal.clear();
            carmans.erase(carmans.begin()+1,carmans.end());
        }
    }
    //facing更正
    if(facing>0 && point2d.size()==4){
        for(int a=0;a<4;a++){
            if(point2d[a].x<= bbox.x-500 || point2d[a].x>= bbox.x+bbox.width+500){
                facing=0;
                point2d.clear();
                pos3dGlobal.clear();
                carmans.erase(carmans.begin()+1,carmans.end());
                break;
            }
            else if(point2d[a].y<= bbox.y-500 || point2d[a].y>= bbox.y+bbox.height+500){
                facing=0;
                point2d.clear();
                pos3dGlobal.clear();
                carmans.erase(carmans.begin()+1,carmans.end());
                break;
            }
        }
    }
}

void Armor::updateNew(ArmorTemp& other,std::vector<float> control){
    //score update
    score=other.score*0.2f+score*0.8f;
    std::vector<float> bboxData0=bbox2CarmanData(other.bbox);
    // kf更新bbox
    std::vector<float> bboxCorrected=carmans[0].correct(bboxData0,control,other.score,videoFGTime+delayTime);
    bbox=carmanData2Bbox(bboxCorrected);
    if(other.haveFacing){
        if(facing<=0){
            // create new (no safty check)
            if(useFast || !other.havePose3d){
                // create 2d
                for(int a=0;a<4;a++){
                    carmans.emplace_back(videoFGTime,2,2,2,0.002);
                    std::vector<float> pointData=point2d2CarmanData(other.point2d[a]);
                    carmans.back().initialize(pointData);
                }
                
            }
            else{
                for (int a = 0; a < 4; a++){
                    carmans.emplace_back(videoFGTime, 2, 3);
                    std::vector<float> pointData = point3d2CarmanData(other.pos3d[a]);
                    carmans.back().initialize(pointData);
                }
            }
        }
        else{
            // update
            if (useFast || !other.havePose3d){
                point2d.clear();
                for (int a = 1; a < carmans.size(); a++){
                    std::vector<float> pointData=point2d2CarmanData(other.point2d[a-1]);
                    std::vector<float> correctedData = carmans[a].correct(pointData, control, videoFGTime);
                    point2d.push_back(carmanData2Point2d(correctedData));
                }
            }
            else{
                for (int a = 1; a < carmans.size(); a++){
                    std::vector<float> pointData = point3d2CarmanData(other.pos3d[a-1]);
                    std::vector<float> correctedData = carmans[a].correct(pointData, videoFGTime);
                    pos3dGlobal[a-1]=carmanData2Point3d(correctedData);
                    //解算2d点不在这里
                }
            }
        }
        facing+=facing<facingMaximum?0.3:0;
    }
    else{
        facing--;
        if(facing<0){
            facing=0;
            point2d.clear();
            pos3dGlobal.clear();
            carmans.erase(carmans.begin()+1,carmans.end());
        }
    }
    //facing更正
    if(facing>0 && point2d.size()==4){
        for(int a=0;a<4;a++){
            if(point2d[a].x<= bbox.x-500 || point2d[a].x>= bbox.x+bbox.width+500){
                facing=0;
                point2d.clear();
                pos3dGlobal.clear();
                carmans.erase(carmans.begin()+1,carmans.end());
                break;
            }
            else if(point2d[a].y<= bbox.y-500 || point2d[a].y>= bbox.y+bbox.height+500){
                facing=0;
                point2d.clear();
                pos3dGlobal.clear();
                carmans.erase(carmans.begin()+1,carmans.end());
                break;
            }
        }
    }
}

void Armor::updateId(int id){
    ids[id]++;
    int idm=0;
    for(int a=0;a<9;a++){
        if(ids[a]>ids[idm]){
            idm=a;
        }
    }
    this->id=idm;
}

void Armor::updateDistance(float dist, float noise){
    if(dist==-1){
        // no update
        if(distance!=-1)
            distCarman.predict();
        return;
    }
    std::vector<float> tmpdist({dist});
    if(distance==-1){
        distance=dist;
        distCarman.initialize(tmpdist);
        return;
    }
    tmpdist=distCarman.correct(tmpdist,noise);
    if(tmpdist.size()>0){
        distance=tmpdist[0];
        return;
    }
    distance=dist;
}

void Armor::lost(){
    // 吃的完有奖励，吃不完有惩罚
    if(facing>0)facing--;
    score-=0.4f;// decay
    std::vector<float> kfbbox = carmans[0].correct();// bbox
    bbox=carmanData2Bbox(kfbbox);
    if(facing<=0 && carmans.size()>1){
        carmans.erase(carmans.begin()+1,carmans.end());
        // **************暂时性保留距离信息*******************
    }
    for(int a=1;a<carmans.size();a++){
        std::vector<float> kfdata = carmans[a].correct();
        if(facing>0){
            if(useFast || pos3dGlobal.size()!=4){
                point2d[a-1]=carmanData2Point2d(kfdata);
            }
            else{
                pos3dGlobal[a-1]=carmanData2Point3d(kfdata);
            }
        }
    }
    if(distance!=-1){
        std::vector<float>tmpdist= distCarman.correct();
        if(tmpdist.size()>0){
            distance=tmpdist[0];
        }
    }
}

void Armor::printBbox(cv::Mat& img){
    cv::rectangle(img,bbox,cv::Scalar(0,100,200),2);
    cv::putText(img,std::to_string(id),cv::Point(bbox.x,bbox.y),cv::FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,100,200),1);
}

std::vector<float> Armor::bbox2CarmanData()const{
    std::vector<float> data;
    data.push_back(bbox.x+0.5f*bbox.width);
    data.push_back(bbox.y+0.5f*bbox.height);
    data.push_back(bbox.width);
    data.push_back(bbox.height);
    return data;
}

std::vector<float> Armor::bbox2CarmanData(cv::Rect ip) const{
    std::vector<float> data;
    data.push_back(ip.x+0.5f*ip.width);
    data.push_back(ip.y+0.5f*ip.height);
    data.push_back(ip.width);
    data.push_back(ip.height);
    return data;
}

cv::Rect Armor::carmanData2Bbox(std::vector<float> data)const{
    cv::Rect bbox;
    bbox.x = data[0]-0.5f*data[2];
    bbox.y = data[1]-0.5f*data[3];
    bbox.width = data[2];
    bbox.height = data[3];
    return bbox;
}

std::vector<float> Armor::point3d2CarmanData(cv::Point3f p)const{
    std::vector<float> data;
    data.push_back(p.x);
    data.push_back(p.y);
    data.push_back(p.z);
    return data;
}

cv::Point3f Armor::carmanData2Point3d(std::vector<float> data)const{
    cv::Point3f p;
    p.x = data[0];
    p.y = data[1];
    p.z = data[2];
    return p;
}

cv::Point Armor::carmanData2Point2d(std::vector<float> data)const{
    cv::Point p;
    p.x = data[0];
    p.y = data[1];
    return p;
}

std::vector<float> Armor::point2d2CarmanData(cv::Point p)const{
    std::vector<float> data;
    data.push_back(p.x);
    data.push_back(p.y);
    return data;
}

RTVec Armor::carmanData2Rtvec(std::vector<float> data) const{
    RTVec op;
    op.tvec.push_back(data[0]);
    op.tvec.push_back(data[1]);
    op.tvec.push_back(data[2]);
    op.rvec.push_back(data[3]);
    op.rvec.push_back(data[4]);
    op.rvec.push_back(data[5]);
    return op;
}

std::vector<float> Armor::rtvec2CarmanData(RTVec rtvec) const{
    std::vector<float> data;
    data.push_back(rtvec.tvec[0]);
    data.push_back(rtvec.tvec[1]);
    data.push_back(rtvec.tvec[2]);
    data.push_back(rtvec.rvec[0]);
    data.push_back(rtvec.rvec[1]);
    data.push_back(rtvec.rvec[2]);
    return data;
}