#include<opencv2/opencv.hpp>
#include<vector>

//mapInfo:
// 1250*1250  25*25block
// 1*1block=50*50pixel
// resized: 200*200,8pix
// roomFinder: gauss 140*140 conv center
// real :10.4---11.71---13.71，正反相等
// 也就是1格2米，0.35车宽

// 计划路径点
class Project{
public:
    std::vector<cv::Point2f> path;// 顺序存储，scale=real
    cv::Rect2f targetArea;// scale=real

    void simplify(float blocksize=0.5);// 简化路径
    void print(cv::Mat& map,float ratio=25.f);
    void printSteadily(std::string winname, cv::Mat& map,float ratio=25.f,int timestep_ms=60);// 逐过程显示路径
    float getLength();
    std::vector<cv::Point2f> mapTo(float ratio)const;// 映射到现实坐标系
    cv::Point2f mapAt(int index,float ratio);// 映射到现实坐标系
    float getApproxTime()const;// 预计用时
};

// 表示操作的集合
struct MovementInfo{
    float xA=0,yA=0;// xy方向的加速度，一般都是+/-的最大加速度
    MovementInfo(MovementInfo& other){
        xA=other.xA;yA=other.yA;
    }
    MovementInfo(){
        xA=0;yA=0;
    }
    MovementInfo& operator=(MovementInfo& other){
        xA=other.xA;yA=other.yA;
        return *this;
    }
    MovementInfo& operator=(MovementInfo other){
        xA=other.xA;yA=other.yA;
        return *this;
    }
};


class Navigator{
public:
    float maxA;// 最大加速度
    float maxspeed;// 最大速度
    int carsize;// 车辆尺寸(最好向上取整)
    int blocksize;// 地图块尺寸
    int mapsize;// 地图块数量
    cv::Point2f lastPos;// 上次位置
    bool haveInfo;// 是否有信息
    
    Navigator();
    Navigator(cv::Mat& map);
    void setMap(cv::Mat& map);// 输入为8uc1图片，墙黑路白
    
    std::vector<cv::Point> getTarget();// 获取目标点，排序为曼哈顿距离（被代替）
    std::vector<cv::Rect2f> getTargetArea();// 获取目标区域，排序为曼哈顿距离
    std::vector<Project> getBestRoute(cv::Point start=cv::Point(1,1));// 一步到位寻找最优，on2的复杂度
    
    Project search(cv::Point start, cv::Point end,bool useHeuristic=true);// 寻路
    MovementInfo navi(Project& route,int& index ,cv::Point2f carPos, float xspeed,float yspeed,bool topGear=true);// 导航
private:
    cv::Mat map;// 地图
    cv::Mat costMap;// 寻路辅助地图
    bool naviMutex;// using
    void graphSimplify(); // 地图简化，原图直出，精确度很重要
    cv::Point mapBack(cv::Point p)const;// 简化图转化为原图坐标

    struct Node{
        int x, y;
        float cost, estim;
        int parentIndex;//-1=无父节点
        Node(int x, int y) : x(x), y(y), cost(0), estim(0), parentIndex(-1) {}
        Node(cv::Point p) : x(p.x), y(p.y), cost(0), estim(0), parentIndex(-1) {}
        Node(cv::Point p, float _cost, float _estim, int parentIndex = -1) : x(p.x), y(p.y), cost(_cost), estim(_estim), parentIndex(parentIndex) {}
        inline float getPrice()const{return cost+estim;}
        //可能需要重载operator=
        inline Node operator=(const Node &n){
            x=n.x;y=n.y;cost=n.cost;estim=n.estim;parentIndex=n.parentIndex;
            return *this;
        }

    };
    struct CompareNode{
        bool operator()(const Node &n1, const Node &n2) const{
            // 小于
            return n1.getPrice() > n2.getPrice(); // 反向比较用于 priority_queue
        }
    };


    MovementInfo naviF(Project &route, int &index, cv::Point2f carPos, float xspeed, float yspeed); // 快速导航
    MovementInfo naviS(Project &route, int &index, cv::Point2f carPos, float xspeed, float yspeed); // 普通导航

    float getDistanceR(cv::Point a,cv::Point b)const;// 欧几里得距离
    float getDistanceF(cv::Point a,cv::Point b)const;// 曼哈顿距离
};