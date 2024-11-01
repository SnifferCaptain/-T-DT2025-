#include"InferCore.h"
#include<openvino/runtime/properties.hpp>
#include<openvino/runtime/intel_cpu/properties.hpp>
#include<openvino/pass/pass.hpp>
#include<ostream>
#include<algorithm>

cv::Rect Result::toCvRect(int imgw,int imgh){
    return cv::Rect(x*imgw-imgw*w*0.5f,y*imgh-imgh*h*0.5f,w*imgw,h*imgh);
}

void Result::draw(cv::Mat& img){
    srand(class_id);
    cv::Scalar color(rand()%255,rand()%255,rand()%255);
    cv::rectangle(img,toCvRect(img.cols,img.rows),color,3);
}

InferCore::InferCore(){
    avaliable=false;
    confThs=0.4f;
    nmsThs=0.2f;
}

InferCore::InferCore(std::string model_path,bool optimize,std::string idDetect){
    avaliable=load(model_path,optimize,idDetect);
    confThs=0.4f;
    nmsThs=0.2f;
}

bool InferCore::load(std::string model_path, bool optimize,std::string idDetect){
    confThs=0.2f;
    try{
        core.reset(new ov::Core());
        if (optimize)
            this->optimizeCore();
        model = core->compile_model(model_path, "CPU");
        std::cout<<"模型加载完毕"<<std::endl;
        request= model.create_infer_request();
        std::cout<<"请求创建完毕"<<std::endl;
        auto tensori = request.get_tensor("images");
        inputc = tensori.get_shape()[1]; // 3
        inputh = tensori.get_shape()[2];  // 640
        inputw = tensori.get_shape()[3];   // 640
        auto tensoro = request.get_tensor("output0");
        outputh = tensoro.get_shape()[1]; // 7
        outputw = tensoro.get_shape()[2]; // 8400
        int classesNum = tensoro.get_shape()[1] - 4;
        //for (int i = 0; i < classesNum; i++){
        //    colors.push_back(cv::Scalar(rand()%255,rand()%255,rand()%255));
        //}
        colors.push_back(cv::Scalar(0,0,255));
        colors.push_back(cv::Scalar(0,255,0));
        colors.push_back(cv::Scalar(255,0,0));
        std::cout<<"图像模型成功加载"<<std::endl;

    }catch(const ov::Exception& e){
        std::cout<<"完啦！！！！模型完啦！！！"<<std::endl;
        std::cout << e.what();
        return false;
    }
    try{
        if (!idDetect.empty()){
            idmodel = core->compile_model(idDetect, "CPU");
            idrequest = idmodel.create_infer_request();
            idAvaliable=true;
            std::cout<<"idDetect模型加载成功"<<std::endl;
        }
        else{
            std::cout<<"不加载idDetect模型"<<std::endl;
        }
    }catch(const ov::Exception& e){
        std::cout<<"还好！还有救！！"<<std::endl;
        std::cout << e.what();
    }
    std::cout<<"全部加载完毕"<<std::endl;
    return true;
}

void InferCore::optimizeCore(){
    //CPU
    core->set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    core->set_property(ov::hint::inference_precision(ov::element::bf16));
    core->set_property(ov::enable_profiling(false));
    core->set_property(ov::inference_num_threads(20));               // 线程池 8p4e
    core->set_property(ov::streams::num(ov::streams::NUMA));         // ov::streams::AUTO));// 线程池
    core->set_property(ov::hint::enable_cpu_pinning(true));          // 线程固定到numa，减少ipc
    core->set_property(ov::intel_cpu::denormals_optimization(true)); // 浮点运算精度优化

    core->set_property(ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    core->set_property(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY));
    //core->set_property((ov::hint::dynamic_quantization_group_size(1024)));//原始配置就是最佳的
    core->set_property(ov::force_tbb_terminate(true));// tbb线程池
    // core->set_property(ov::hint::enable_hyper_threading(true));    //原始配置最好,别展开
    // core->set_property("CPU",ov::hint::model_priority(ov::hint::Priority::HIGH));//敢开就敢完蛋

    auto pin=core->get_property("CPU",ov::supported_properties);
    std::cout<<"================CPU支持的属性=============="<<std::endl;
    for(auto& p:pin){
        std::string str=core->get_property("CPU",p).as<std::string>();
        std::cout << p << "  :  "<<str<<std::endl;
    }
    //GPU
    //没有GPU
}

std::vector<Result> InferCore::infer(cv::Mat& img){
    if(!avaliable)return std::vector<Result>();
    cv::Mat inputMat=fillContour(img);
    fillTensor(inputMat);
    request.infer();
    std::vector<Result> res= afterProcess();
    decodeResult(img, res);
    return res;
}

int InferCore::inferId(cv::Mat& img){
    if(idAvaliable==false)return 0;
    cv::Mat inputMat;
    cv::resize(img,inputMat,cv::Size(20,28));
    cv::cvtColor(inputMat,inputMat,cv::COLOR_BGR2GRAY);
    inputMat.convertTo(inputMat,CV_32F,1.f/255.f);
    float *ip = idrequest.get_tensor("input").data<float>();
    int t = 0;
    for (; t < 560; t++){
        float cur=inputMat.at<float>(t);
        ip[t] = cur;
    }
    idrequest.infer();
    //cv::resize(inputMat,inputMat,cv::Size(200,280));
    //inputMat.convertTo(inputMat,CV_8UC1,255.);
    //cv::imshow("id",inputMat);
    //cv::waitKey(0);
    float *op = idrequest.get_tensor("output").data<float>();
    int max = 0;
    for (int i = 0; i < 9; i++){
        float nnn = op[i];
        if (op[i] > op[max])
            max = i;
    }
    std::cout<<max<<std::endl;
    return max;
}

cv::Mat InferCore::fillContour(cv::Mat img){
    // 不改变channel顺序
    int targetW=request.get_tensor("images").get_shape()[3];
    int targetH=request.get_tensor("images").get_shape()[2];
    cv::Mat op(targetH,targetW,CV_8UC3,cv::Scalar(114,114,114));
    float wdh = static_cast<float>(img.cols) / static_cast<float>(img.rows);
    float twdh = static_cast<float>(targetW) / static_cast<float>(targetH);
    if (wdh > twdh)
    {
        // 等比例resize后，填充上下侧为黑色
        cv::resize(img, img, cv::Size(targetW, img.rows * targetW / img.cols));
        cv::Rect roi(0,(targetH-img.rows*targetW/img.cols)/2,targetW,img.rows*targetW/img.cols);
        img.copyTo(op(roi));
        return op;
    }
    else
    {
        // 等比例resize后，填充左右侧为黑色
        cv::resize(img, img, cv::Size(img.cols * targetH / img.rows, targetH));
        cv::Rect roi((targetW-img.cols*targetH/img.rows)/2,0,img.cols*targetH/img.rows,targetH);
        img.copyTo(op(roi));
        return op;
    }
}

void InferCore::fillTensor(cv::Mat& img){
    //NOTE: 输入张量形状为[1,3,640,640],             !!!!R!!!!G!!!!B!!!!
    auto tensor=request.get_tensor("images");
    int channel=tensor.get_shape()[1];//3
    int height=tensor.get_shape()[2];//640
    int width=tensor.get_shape()[3];//640
    int hxw = height * width;
    cv::Mat img_resize;
    cv::resize(img, img_resize, cv::Size(width, height));
    img_resize.convertTo(img_resize, CV_32FC3,1./255.);
    
    if(tensor.get_element_type()==ov::element::f32){
        float *data = tensor.data<float>();
        // HWC 2 CHW
        for (int i = 0; i < channel; i++){
            for (int j = 0; j < height; j++){
                for (int k = 0; k < width; k++){
                    data[i * hxw + j * width + k] = img_resize.at<cv::Vec3f>(j, k)[abs(i-channel+1)];
                }
            }
        }
    }
    else if(tensor.get_element_type()==ov::element::f16){
        ov::float16 *data = tensor.data<ov::float16>();
        for (int i = 0; i < channel; i++){
            for (int j = 0; j < height; j++){
                for (int k = 0; k < width; k++){
                    data[i * hxw + j * width + k] = img_resize.at<cv::Vec3f>(j, k)[abs(i-channel+1)];
                }
            }
        }
    }
}

std::vector<Result> InferCore::afterProcess(){
    //NOTE: shape[1,4+3,8400]
    auto tensor=request.get_tensor("output0");
    int boxSize = tensor.get_shape()[2];//8400
    int infoSize = tensor.get_shape()[1];//7
    std::cout<<"boxSize:"<<boxSize<<"  infoSize:"<<infoSize<<std::endl;
    std::vector<Result> result;
    if(tensor.get_element_type()==ov::element::f32){
        float *data = tensor.data<float>(); //[7,8400]
        // opencv 就是屎
        for (int a = 0; a < boxSize; a++){
            int scoreMax = 4;
            for (int b = 4; b < infoSize; b++){
                float now = valueAt<float>(data, a, b, boxSize);
                float max = valueAt<float>(data, a, scoreMax, boxSize);
                if (now > max){
                    scoreMax = b;
                }
            }
            float score = valueAt<float>(data, a, scoreMax, boxSize);
            if (score < confThs)continue;
            Result r;
            r.x = valueAt(data, a, 0, boxSize) / static_cast<float>(inputw);
            r.y = valueAt(data, a, 1, boxSize) / static_cast<float>(inputh);
            r.w = valueAt(data, a, 2, boxSize) / static_cast<float>(inputw);
            r.h = valueAt(data, a, 3, boxSize) / static_cast<float>(inputh);
            r.conf = score;
            r.class_id = scoreMax - 4;
            result.push_back(r);
        }
    }
    else if(tensor.get_element_type()==ov::element::f16){
        ov::float16 *data = tensor.data<ov::float16>(); //[7,8400] alignas(16)16位对齐
        // opencv 就是屎
        for(int a=0;a<boxSize;a++){
            int scoreMax=4;
            float score=0.f;
            for(int b=4;b<infoSize;b++){
                float now=valueAt<ov::float16>(data,a,b,boxSize);
                if(now>score){
                    score=now;
                    scoreMax=b;
                }
            }
            if(score<confThs)continue;
            Result r;
            r.x=valueAt(data, a, 0, boxSize)/static_cast<float>(inputw);
            r.y=valueAt(data, a, 1, boxSize)/static_cast<float>(inputh);
            r.w=valueAt(data, a, 2, boxSize)/static_cast<float>(inputw);
            r.h=valueAt(data, a, 3, boxSize)/static_cast<float>(inputh);
            r.conf=score;
            r.class_id=scoreMax-4;
            result.push_back(r);
        }
    }
    nms(result,nmsThs);
    //安全检查
    for(auto& r:result){
        r.w = std::min({r.w, r.x * 2.f, (1.f - r.x) * 2.f});
        r.h = std::min({r.h, r.y * 2.f, (1.f - r.y) * 2.f});
    }
    return result;
}

void InferCore::decodeResult(cv::Mat& img,std::vector<Result>& result){
    int& originW=img.cols;
    int& originH=img.rows;
    int targetW=request.get_tensor("images").get_shape()[3];
    int targetH=request.get_tensor("images").get_shape()[2];
    float wdh = static_cast<float>(originW) / static_cast<float>(originH);
    float twdh = static_cast<float>(targetW) / static_cast<float>(targetH);
    if (wdh > twdh){
        // 填充上下侧
        for (size_t i = 0; i < result.size(); i++) {
            result[i].y = (result[i].y * (1/twdh)-((1/twdh) - (1 / wdh)) / 2) / (1 / wdh);// 对了！
            result[i].h = result[i].h * wdh / twdh;// 这个是对的
            // 越界检测
            float ymin = result[i].y - result[i].h / 2;
            float ymax = result[i].y + result[i].h / 2;
            if (ymin < 0 || ymax>1 ) {
                result.erase(result.begin() + i);
            }
        }
    }
    else{
        for (size_t i = 0; i < result.size(); i++){
            result[i].x = (result[i].x * twdh - (twdh - wdh) / 2) / wdh;
            result[i].w = result[i].w * twdh / wdh;
            // 越界检测
            float xmin = result[i].x - result[i].w / 2;
            float xmax = result[i].x + result[i].w / 2;
            if (xmin < 0 || xmax > 1){
                result.erase(result.begin() + i);
            }
        }
    }
}

void InferCore::postResult(cv::Mat& img,std::vector<Result>& result,bool show){
    for(auto& r:result){
        cv::rectangle(img,r.toCvRect(img.cols,img.rows),colors[r.class_id],2);
    }
    if(show){
        cv::imshow("result",img);
        //cv::waitKey(0);
    }
}

template<typename T>
inline T& InferCore::valueAt(T* data,int x,int y,int colsize){
    return data[y*colsize+x];
}

void InferCore::nms(std::vector<Result> &results, float iou){
    int a = 0;
    for (; a < results.size(); a++){
        int b = a + 1;
        for (; b < results.size(); b++){
            if (results[a].class_id != results[b].class_id)
                continue;
            if (iou < getIou(results[a],results[b])){
                // yy
                if (results[a].conf > results[b].conf){
                    results.erase(results.begin()+b);
                    b--;
                }
                else{
                    results.erase(results.begin()+a);
                    a--;
                    break;
                }
            }
        }
    }
}

inline float getIou(Result& a,Result& b){
    float a1x=a.x-a.w*0.5f;
    float a1y=a.y-a.h*0.5f;
    float a2x=a.x+a.w*0.5f;
    float a2y=a.y+a.h*0.5f;
    float b1x=b.x-b.w*0.5f;
    float b1y=b.y-b.h*0.5f;
    float b2x=b.x+b.w*0.5f;
    float b2y=b.y+b.h*0.5f;
    float &fromx = a1x > b1x ? a1x : b1x;
    float &fromy = a1y > b1y ? a1y : b1y;
    float &tox = a2x < b2x ? a2x : b2x;
    float &toy = a2y < b2y ? a2y : b2y;

    float w = tox-fromx, h=toy-fromy;
    if (w <= 0 || h <= 0)
    {
        return 0.f;
    }
    float uarea = w * h;
    return uarea / (a.w*a.h+b.w*b.h - uarea);
}

int Imain(){
    InferCore InferCore("../armor3-1300.onnx",false);
    InferCore.confThs=0.5f;
    cv::VideoCapture cap("/home/mrye/Desktop/TDT/T-DT2023-OpenCV/task1/build/Infantry_red.avi");
    cv::Mat frame;
    time_t t0=time(0);
    float avgfps=0.f;
    float frameCount=0;
    while(cap.read(frame)){
        //cv::resize(frame,frame,cv::Size(640,640));
        std::vector<Result> result=InferCore.infer(frame);
        InferCore.postResult(frame,result,true);
        auto key=cv::waitKey(1);
        if(key==27)break;
        //e暂停 不能用于记帧
        else if(key=='e')cv::waitKey(0);

        // fps
        frameCount++;
        time_t t1=time(0);
        if(t1-t0>=1){
            avgfps = float(frameCount) / (t1 - t0);
            std::cout << "fps:" << avgfps << std::endl;
            frameCount=0;
            t0=t1;
        }
    }
    return 0;
}
