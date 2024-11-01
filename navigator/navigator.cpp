#include"navigator.hpp"

void Project::simplify(float blockSize){
    if(path.size()<=1){
        return;
    }
    cv::approxPolyDP(path,path,0.5*blockSize,false);
}

void Project::print(cv::Mat& map,float ratio){
    for(int i=0;i<path.size()-1;i++){
        cv::line(map,path[i]*ratio,path[i+1]*ratio,cv::Scalar(200),2);
    }
}

void Project::printSteadily(std::string winname, cv::Mat& map,float ratio ,int t){
    cv::Scalar color(rand()%255);
    for(int i=0;i<path.size()-1;i++){
        cv::line(map,path[i]*ratio,path[i+1]*ratio,color,2);
        cv::imshow(winname,map);
        cv::waitKey(t);
    }
}

std::vector<cv::Point2f> Project::mapTo(float ratio)const{
    std::vector<cv::Point2f> op;
    for(int i=0;i<path.size();i++){
        op.emplace_back(cv::Point2f(path[i].x*ratio,path[i].y*ratio));
    }
    return op;
}

cv::Point2f Project::mapAt(int index,float ratio){
    cv::Point2f op=path[index];
    op.x*=ratio;
    op.y*=ratio;
    return op;
}

float Project::getLength(){
    float op=cv::arcLength(path,false);
    return op;
}

float Project::getApproxTime()const{
    float op=cv::arcLength(path,false)*0.185f+path.size()*1.134f;
    return op;
}

////////////////////////navi////////////////////////

Navigator::Navigator(){
    carsize = 5;    // 待测
    blocksize = 50; // 25*25,8p
    mapsize = 25;
    maxA = 5.f;// ok
    maxspeed = 5.f;
    naviMutex=false;
    haveInfo=false;
}

Navigator::Navigator(cv::Mat& mapin){
    haveInfo=false;
    carsize = 5;//待测
    blocksize = 50;// 25*25,8p
    mapsize=25;
    maxA = 5.f;// ok
    maxspeed = 5.f;
    naviMutex=false;
    setMap(mapin);
}

void Navigator::setMap(cv::Mat& mapin){
    if(haveInfo){
        return;
    }
    haveInfo=true;
    this->map = mapin;
    graphSimplify();
    //碰撞箱膨胀
    costMap=map.clone();
    cv::bitwise_not(costMap,costMap);
    //cv::imshow("costMap0",costMap);
    costMap.convertTo(costMap,CV_32F);// 0~255
    // 将障碍物设置为无穷大
    for(int i=0;i<costMap.rows;i++){
        for(int j=0;j<costMap.cols;j++){
            if (costMap.at<float>(i, j) >= 254.f)
                costMap.at<float>(i,j)=1000000.f;
            else if (costMap.at<float>(i, j) <= 0.01){
                costMap.at<float>(i,j)=1.f;
            }
        }
    }
    cv::Mat costTemp=costMap.clone();
    cv::resize(costTemp,costTemp,cv::Size(1000,1000));
    //cv::imshow("costMap",costTemp);
    //cv::waitKey(0);
}

float Navigator::getDistanceR(cv::Point p1, cv::Point p2)const{
    return cv::norm(p1-p2);
}

float Navigator::getDistanceF(cv::Point p1, cv::Point p2)const{
    return abs(p1.x-p2.x)+abs(p1.y-p2.y);
}

std::vector<cv::Point> Navigator::getTarget(){
    std::vector<cv::Point> op;
    cv::Mat finder=map.clone();
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::erode(finder,finder,kernel);
    for(int i=0;i<finder.rows;i++){
        for(int j=0;j<finder.cols;j++){
            if(finder.at<uchar>(i,j)>0){
                op.emplace_back(j,i);
                cv::floodFill(finder,cv::Point(j,i),cv::Scalar(0));
            }
        }
    }
    return op;
}

std::vector<cv::Rect2f> Navigator::getTargetArea(){
    std::vector<cv::Rect2f> op;
    cv::Mat finder=map.clone();
    cv::Mat kernel=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    cv::erode(finder,finder,kernel);
    cv::dilate(finder,finder,kernel);
    for(int i=0;i<finder.rows;i++){
        for(int j=0;j<finder.cols;j++){
            if(finder.at<uchar>(i,j)>0){
                int w=0,h=0;
                for(int m=i+1;m<finder.rows;m++){
                    if(finder.at<uchar>(m,j)==0){
                        h=m-i;
                        break;
                    }
                }
                for(int m=j+1;m<finder.cols;m++){
                    if(finder.at<uchar>(i,m)==0){
                        w=m-j;
                        break;
                    }
                }
                cv::Rect2f rect(j,i,w,h);
                op.push_back(rect);
                cv::floodFill(finder,cv::Point(j,i),cv::Scalar(0));
            }
        }
    }
    return op;
}

// 计算给定路径的总距离
int calculatePathCost(const cv::Mat& dist, std::vector<int>& path) {
    int cost = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        cost += dist.at<float>(path[i], path[i + 1]);
    }
    // 返回起点
    cost += dist.at<float>(path.size()-1, 0);
    return cost;
}

// 暴力算法求解TSP
std::vector<int> tspBruteForce(cv::Mat& dist) {
    int n = dist.cols;
    std::vector<int> nodes(n);
    for (int i = 0; i < n; ++i) {
        nodes[i] = i;
    }

    int minCost = INT_MAX;
    std::vector<int> bestPath;

    do {
        int currentCost = calculatePathCost(dist, nodes);
        if (currentCost < minCost) {
            minCost = currentCost;
            bestPath = nodes;
        }
    } while (next_permutation(nodes.begin() + 1, nodes.end()));

    return bestPath;
}

std::vector<Project> Navigator::getBestRoute(cv::Point start){
    //std::vector<cv::Point> targets=getTarget();//old
    std::vector<Project> op;
    std::vector<cv::Rect2f> ta=getTargetArea();
    std::vector<cv::Point> targets;
    for(int i=0;i<ta.size();i++){
        targets.push_back(cv::Point(ta[i].x+ta[i].width/2,ta[i].y+ta[i].height/2));
    }
    targets.insert(targets.begin(),start);
    std::vector<std::vector<Project>> routes;// 0=起始点,
    
    for(int a=0;a<targets.size();a++){
        std::vector<Project> temp;
        for(int b=a+1;b<targets.size();b++){
            temp.push_back(search(targets[a],targets[b],true));
        }
        routes.push_back(temp);
    }
    // 54321,相加表达
    cv::Mat costList=cv::Mat::zeros(targets.size(),targets.size(),CV_32F);
    for(int a=0;a<routes.size();a++){
        for(int b=0;b<routes[a].size();b++){
            costList.at<float>(a,b+a+1)=routes[a][b].getLength();
            costList.at<float>(b+a+1,a)=costList.at<float>(a,b+a+1);
        }
    }
    std::vector<int> best=tspBruteForce(costList);
    for(int i=0;i<best.size()-1;i++){
        if(best[i]<best[i+1]){
            op.push_back(routes[best[i]][best[i + 1] - best[i] - 1]);
            op.back().targetArea=ta[best[i+1]-1];
        }
        else{
            // 重新生成，要不会反（最懒的方法）
            op.push_back(search(targets[best[i]],targets[best[i+1]]));
            op.back().targetArea=ta[best[i+1]-1];
        }
    }
    // map to real
    float ratio = 2.f / static_cast<float>(blocksize);
    for(int i=0;i<op.size();i++){
        cv::Point2f areaCenter = (op[i].targetArea.tl() + op[i].targetArea.br()) / 2.f;
        for(int j=0;j<op[i].path.size();j++){
            op[i].path[j]=op[i].mapAt(j,ratio);
            if(op[i].targetArea.contains(op[i].path[j]) && cv::norm(areaCenter-op[i].path[j])<2.f){
                op[i].path.erase(op[i].path.begin()+j);
                j--;
            }
        }
        if(i!=0){
            areaCenter = (op[i-1].targetArea.tl() + op[i-1].targetArea.br()) / 2.f;// 上一个终点
            int maxindex=0;
            for (int j = 0; j < op[i].path.size(); j++){
                if (op[i - 1].targetArea.contains(op[i].path[j])){
                    maxindex=j;
                }
                /*
                //old
                    if (op[i-1].targetArea.contains(op[i].path[j]) && cv::norm(areaCenter - op[i].path[j]) < 2.5f){
                    op[i].path.erase(op[i].path.begin() + j);
                    j--;
                }
                */
            }
            op[i].path.erase(op[i].path.begin(),op[i].path.begin()+maxindex-1);
        }
        op[i].simplify();
        std::cout<<" targetArea: "<< op[i].targetArea <<"  routeLen:"<< op[i].getLength()<<"  routeCorner: "<< op[i].path.size() <<"approxTime: "<< op[i].getApproxTime() <<std::endl;
    }
    return op;
}

Project Navigator::search(cv::Point start, cv::Point end,bool useHeuristic){
    Project op;
    auto& result=op.path;
    //使用a*寻路
    std::priority_queue<Node,std::vector<Node>,CompareNode> open;// 表示待探索的节点
    std::vector<std::vector<bool>> closed(map.rows, std::vector<bool>(map.cols, false)); // 表示已探索的节点
    std::vector<Node> all;// 存储所有节点
    if(useHeuristic){
        Node node(start,0,getDistanceF(start,end));
        open.push(node);
        all.push_back(node);
    }
    else{
        Node node(start, 0, getDistanceR(start, end));
        open.push(node);
        all.push_back(node);
    }
    while(!open.empty()){
        Node node=open.top();
        int nodeIndex=all.size();
        open.pop();
        cv::Point p(node.x,node.y);
        if(p==end){
            // 找到路径
            std::vector<cv::Point> revOp; // 逆序
            result.clear();
            while(node.parentIndex!=-1){
                revOp.emplace_back(cv::Point2f(node.x, node.y));
                node = all[node.parentIndex];
            }
            revOp.emplace_back(node.x, node.y);

            for(int a=revOp.size()-1;a>0;a--){//特意漏掉最后1个
                result.push_back(mapBack(revOp[a]));
            }
            return op;
        }
        // 注册节点
        all.push_back(node);
        closed[p.y][p.x] = true;
        // 遍历邻居
        int dx[] = {-1, 0, 1, 0};
        int dy[] = {0, -1, 0, 1};

        for (int i = 0; i < 4; ++i) {
            cv::Point neighbor(p.x + dx[i], p.y + dy[i]);

            if (neighbor.x < 0 || neighbor.x >= map.cols || neighbor.y < 0 || neighbor.y >= map.rows) continue;
            if (closed[neighbor.y][neighbor.x]) continue;
            //if (map.at<uchar>(neighbor) == 0) continue; // 墙壁

            float newCost = node.cost + costMap.at<float>(neighbor);
            float newEstim = useHeuristic?getDistanceF(neighbor,end):getDistanceR(neighbor,end);
            open.emplace(neighbor, newCost, newEstim, nodeIndex);

        }
        // debug
        /*
        cv::Mat shower = cv::Mat::zeros(map.size(), CV_8UC1);
        for (int i = 0; i < map.rows; ++i)
        {
            for (int j = 0; j < map.cols; ++j)
            {
                if (closed[i][j])
                    shower.at<uchar>(i, j) = 128;
                else
                    shower.at<uchar>(i, j) = map.at<uchar>(i, j);
            }
        }
        cv::resize(shower, shower, cv::Size(1024, 1024));
        cv::imshow("shower", shower);
        cv::waitKey(1);

        */
    }
    std::cout<<"No path found!"<<std::endl;
    return op;
}

MovementInfo Navigator::navi(Project& route,int& index ,cv::Point2f carPos, float xspeed,float yspeed,bool topGear){
    if(index>=route.path.size() || naviMutex){
        MovementInfo op;
        op.xA=0;
        op.yA=0;
        return op;
    }
    naviMutex=true;
    MovementInfo op;
    //route.simplify();
    //topGear=true;
    if(topGear){
        op = naviF(route, index, carPos, xspeed, yspeed);
    } 
    else{
        op = naviS(route, index, carPos, xspeed, yspeed);
    }
    naviMutex=false;
    return op;
}

MovementInfo Navigator::naviS(Project& route,int& index ,cv::Point2f carPos, float xspeed,float yspeed){
    // 低速导航： 到节点后停止
    // note：防止摇摆，归零加速度范围需要比极限范围大
    MovementInfo op;
    // 瞬移判断
    if(index!=0){
        float distanceTP=cv::norm(lastPos-carPos);
        if(distanceTP>10.f){
            // 抵达终点
            op.xA = 0;
            op.yA = 0;
            index = route.path.size()-1;
            return op;
        }
        lastPos=carPos;
    }
    else{
        lastPos=carPos;
    }
    
    cv::Point target=route.path[index];
    float distanceX=target.x-carPos.x;
    float distanceY=target.y-carPos.y;
    float absDistX=abs(distanceX);
    float absDistY=abs(distanceY);
    float distance=absDistX+absDistY;
    float distanceR = sqrtf(distanceX * distanceX + distanceY * distanceY);
    float distanceEnd=cv::norm((route.targetArea.tl()+route.targetArea.br())/2.f-carPos);
    
    //debug
    std::cout<<"distance:"<<distance<<std::endl;
    std::cout<<"carPos:"<<carPos<<std::endl;
    std::cout<<"target:"<<target<<std::endl;
    std::cout<<"distanceX:"<<distanceX<<std::endl;
    std::cout<<"distanceY:"<<distanceY<<std::endl;
    
    if(distanceEnd<=2.5f){
        // 抵达终点
        op.xA=0;
        op.yA=0;
        index=route.path.size();
        return op;
    }

    if (absDistX <= 0.35f && absDistY<=0.35f && abs(xspeed) <= 1.f && abs(yspeed) <= 1.f){
        // 抵达目标点
        index++;
        if(index==route.path.size()){
            op.xA=0;
            op.yA=0;
            return op;
        }
        target = route.path[index];
        distanceX=target.x-carPos.x;
        distanceY=target.y-carPos.y;
        absDistX = abs(distanceX);
        absDistY = abs(distanceY);
        distance = absDistX + absDistY;
        distanceR = sqrt(distanceX * distanceX + distanceY * distanceY);
    }
    else if (index==0 && absDistX <= 0.5f && absDistY <= 0.5f && abs(xspeed) <= 2.f && abs(yspeed) <= 2.f){
        // 第一个目标点
        index++;
        if (index == route.path.size()){
            op.xA = 0;
            op.yA = 0;
            return op;
        }
        target = route.path[index];
        distanceX = target.x - carPos.x;
        distanceY = target.y - carPos.y;
        absDistX = abs(distanceX);
        absDistY = abs(distanceY);
        distance = absDistX + absDistY;
        distanceR = sqrt(distanceX * distanceX + distanceY * distanceY);
    }
    float periodSize = 2.f; // 1c
    if (0 != index){
        periodSize = cv::norm(route.path[index - 1] - route.path[index]);
    }
    std::cout<<"periodSize:"<<periodSize<<std::endl;
    // 2ax=v^2-0   ==>  x=v^2/2a
    float alertDistanceX=xspeed*xspeed/2.f/maxA+abs(xspeed)*0.075f;// 近距离没问题，远距离有
    float alertDistanceY=yspeed*yspeed/2.f/maxA+abs(yspeed)*0.075f;// 长距离过度补偿
    
    if(distance<5.f){
        // 短距离延迟优化
        alertDistanceX+=0.25f;
        alertDistanceY+=0.25f;
    }
    
    if (periodSize < 7.2f){
        // 短距离延迟优化
        alertDistanceX += 0.15f;
        alertDistanceY += 0.15f;
    }
    if (periodSize < 5.1f){
        alertDistanceX += 0.15f + abs(xspeed) * 0.08f;
        alertDistanceY += 0.15f + abs(yspeed) * 0.08f;
    }
    if (periodSize < 3.f && distanceR > 0.3f){
        //alertDistanceX *= 0.95f;
        //alertDistanceY *= 0.95f;
        alertDistanceX += -0.1f;
        alertDistanceY += -0.1f;
    }

    std::cout << "maxA:" << maxA << std::endl;
    std::cout << "alertDistanceX:" << alertDistanceX << std::endl;
    std::cout << "alertDistanceY:" << alertDistanceY << std::endl;
    std::cout<<"xspeed:"<<xspeed<<std::endl;
    std::cout<<"yspeed:"<<yspeed<<std::endl;

    if(absDistX>alertDistanceX){
        // 加速！
        op.xA = maxspeed * distanceX / distanceR;
    }
    else{
        op.xA = 0;
    }
    
    if(absDistY >alertDistanceY){
        // 加速！
        op.yA = maxspeed * distanceY / distanceR;
    }
    else{
        op.yA = 0;
    }
    if(op.xA==0 && op.yA==0 && abs(xspeed)<0.4f && abs(yspeed)<0.4f && distanceR>0.3f && distanceR<2.f){
        // 错误的停止
        if(periodSize<5.f){
            op.xA = 1.f* distanceX / distanceR;
            op.yA = 1.f* distanceY / distanceR;
        }
        else{
            op.xA = 1.5f * distanceX / distanceR;
            op.yA = 1.5f * distanceY / distanceR;
        }
    }
    return op;
}

MovementInfo Navigator::naviF(Project& route,int& index ,cv::Point2f carPos, float xspeed,float yspeed){
    // 改进：弧线过弯
    MovementInfo op;
    // 瞬移判断
    if(index!=0){
        float distanceTP=cv::norm(lastPos-carPos);
        if(distanceTP>10.f){
            // 抵达终点
            op.xA = 0;
            op.yA = 0;
            index = route.path.size()-1;
            return op;
        }
        lastPos=carPos;
    }
    else{
        lastPos=carPos;
    }
    
    cv::Point target=route.path[index];
    float distanceX=target.x-carPos.x;
    float distanceY=target.y-carPos.y;
    float absDistX=abs(distanceX);
    float absDistY=abs(distanceY);
    float distance=absDistX+absDistY;
    float distanceR = sqrtf(distanceX * distanceX + distanceY * distanceY);
    float distanceEnd=cv::norm((route.targetArea.tl()+route.targetArea.br())/2.f-carPos);
    
    //debug
    std::cout<<"distance:"<<distance<<std::endl;
    std::cout<<"carPos:"<<carPos<<std::endl;
    std::cout<<"target:"<<target<<std::endl;
    std::cout<<"distanceX:"<<distanceX<<std::endl;
    std::cout<<"distanceY:"<<distanceY<<std::endl;
    
    if(distanceEnd<=2.5f){
        // 抵达终点
        op.xA=0;
        op.yA=0;
        index=route.path.size();
        return op;
    }
    

    if(distanceR<=0.45f && abs(xspeed)<=2.f && abs(yspeed)<=2.f){
        // 抵达目标点
        index++;
        if(index==route.path.size()){
            op.xA=0;
            op.yA=0;
            return op;
        }
        target = route.path[index];
        distanceX=target.x-carPos.x;
        distanceY=target.y-carPos.y;
        absDistX = abs(distanceX);
        absDistY = abs(distanceY);
        distance = absDistX + absDistY;
        distanceR = sqrt(distanceX * distanceX + distanceY * distanceY);
    }
    float periodSize = 2.f; // 1c
    if (0 != index){
        periodSize = cv::norm(route.path[index - 1] - route.path[index]);
    }
    std::cout<<"periodSize:"<<periodSize<<std::endl;
    // 2ax=v^2-0   ==>  x=v^2/2a
    float alertDistanceX=xspeed*xspeed/2.f/maxA+abs(xspeed)*0.06f;// 近距离没问题，远距离有
    float alertDistanceY=yspeed*yspeed/2.f/maxA+abs(yspeed)*0.06f;// 长距离过度补偿
    
    if(distance<7.f){
        // 短距离延迟优化
        alertDistanceX+=0.15f;
        alertDistanceY+=0.15f;
    }
    
    if (periodSize < 7.2f){
        // 短距离延迟优化
        alertDistanceX += 0.25f;
        alertDistanceY += 0.25f;
    }
    if (periodSize < 5.2f){
        alertDistanceX += 0.3f + abs(xspeed) * 0.08f;
        alertDistanceY += 0.3f + abs(yspeed) * 0.08f;
    }
    if (periodSize < 3.2f && distanceR > 0.4f){
        //alertDistanceX += 0.05f;
        //alertDistanceY += 0.05f;
    }

    std::cout << "maxA:" << maxA << std::endl;
    std::cout << "alertDistanceX:" << alertDistanceX << std::endl;
    std::cout << "alertDistanceY:" << alertDistanceY << std::endl;
    std::cout<<"xspeed:"<<xspeed<<std::endl;
    std::cout<<"yspeed:"<<yspeed<<std::endl;

    if(absDistX>alertDistanceX){
        // 加速！
        op.xA = maxspeed * distanceX / distanceR;
    }
    else{
        op.xA = 0;
    }
    
    if(absDistY >alertDistanceY){
        // 加速！
        op.yA = maxspeed * distanceY / distanceR;
    }
    else{
        op.yA = 0;
    }
    if(op.xA==0 && op.yA==0 && abs(xspeed)<0.3f && abs(yspeed)<0.3f && distanceR>0.3f){
        // 错误的停止
        if(periodSize<5.f){
            op.xA = 0.5f* distanceX / distanceR;
            op.yA = 0.5f* distanceY / distanceR;
        }
        else{
            op.xA = 1.f * distanceX / distanceR;
            op.yA = 1.f * distanceY / distanceR;
        }
    }
    return op;
}

void Navigator::graphSimplify(){
    // y,x=25*m,25*n m,n<=50
    // 黑色涂黑，白色不管
    cv::Mat simplify=cv::Mat::zeros(mapsize*2+1,mapsize*2+1,CV_8UC1);
    cv::bitwise_not(simplify,simplify);
    for(int a=0;a<=mapsize*2;a++){
        for(int b=0;b<=mapsize*2;b++){
            simplify.at<uchar>(a,b)=map.at<uchar>(blocksize/2*a,blocksize/2*b);
        }
    }

    map=simplify;
    //cv::resize(simplify,simplify,cv::Size(1000,1000));
    //cv::imshow("simplify", simplify);
    // cv::waitKey(0);
}

cv::Point Navigator::mapBack(cv::Point p)const{
    return cv::Point(p.x*blocksize/2,p.y*blocksize/2);
}

int Nmain(){
    //test
    cv::Mat map=cv::imread("../map2.png",cv::IMREAD_GRAYSCALE);
    Navigator navi(map);
    //cv::Point start(1,1);// 起始点
    //cv::Point end(49,49);// 目标点
    //std::vector<cv::Point> target=navi.getTarget();
    //target.insert(target.begin(),cv::Point(1,1));
    //for(int a=0;a<target.size()-1;a++){
    //    Project route = navi.search(target[a], target[a+1]);
    //    route.printSteadily("aha",map);
    //}
    std::vector<Project> routes=navi.getBestRoute();    
    for(auto& route:routes){
        route.printSteadily("aha",map,25.f,100);
    }

    cv::imshow("map",map);
    cv::waitKey(0);
    return 0;
}
