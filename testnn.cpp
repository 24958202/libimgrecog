/*
    g++ -std=c++20 /Users/dengfengji/ronnieji/lib/TorchNN/main/testnn.cpp -o /Users/dengfengji/ronnieji/lib/TorchNN/main/testnn -I/Users/dengfengji/ronnieji/lib/TorchNN/include -I/Users/dengfengji/ronnieji/lib/TorchNN/src /Users/dengfengji/ronnieji/lib/TorchNN/src/*.cpp -I/Users/dengfengji/ronnieji/lib/project/include \
-I/Users/dengfengji/ronnieji/lib/project/src \
-I/opt/homebrew/Cellar/tesseract/5.4.1_1/include \
-L/opt/homebrew/Cellar/tesseract/5.4.1_1/lib -ltesseract \
-I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4 \
-L/opt/homebrew/Cellar/opencv/4.10.0_12/lib \
-Wl,-rpath,/opt/homebrew/Cellar/opencv/4.10.0_12/lib \
-lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_video -lopencv_dnn \
-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 \
-I/opt/homebrew/Cellar/boost/1.86.0_1/include \
-I/opt/homebrew/Cellar/icu4c@76/76.1_1/include \
-L/opt/homebrew/Cellar/icu4c@76/76.1_1/lib -licuuc -licudata \
/opt/homebrew/Cellar/boost/1.86.0_1/lib/libboost_system.a \
/opt/homebrew/Cellar/boost/1.86.0_1/lib/libboost_filesystem.a \
-I$LIBTORCH_PATH/include \
-I$LIBTORCH_PATH/include/torch/csrc/api/include \
-L$LIBTORCH_PATH/lib \
-Wl,-rpath,$LIBTORCH_PATH/lib \
-ltorch -ltorch_cpu -lc10 \
$(pkg-config --cflags --libs sdl2) \
$(pkg-config --cflags --libs sdl2_image) \
$(pkg-config --cflags --libs libjpeg) \
$(pkg-config --cflags --libs libturbojpeg) \
-DOPENCV_VERSION=4.10.0_12 -lexiv2
*/

#include <iostream>
#include <thread>
#include <chrono>
#include "torchnn.h"

int main(){
    torchnn tnn_j;
    /*
        compress the corpus
    */
    tnn_j.img_compress("/Users/dengfengji/ronnieji/Kaggle/imagenet/ILSVRC/Done",18);
    /*
        put all images into a serialized file image.dat
    */
    tnn_j.serializeImages(100,"/Users/dengfengji/ronnieji/Kaggle/imagenet/ILSVRC/Done","/Users/dengfengji/ronnieji/lib/TorchNN/main/image.dat");
    /*
        const int imageSize = 100; // Resize images to 100x100
        const int numEpochs = 10;
        const int batchSize = 16;
        const float learningRate = 1e-3;
    */
    tnn_j.train_model(
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/image.dat",
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/siamese_model.pt",
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/labelMap.txt",
        100,
        10,//numEpochs
        16,//batchSize
        1e-3//learningRate
        );
    std::cout << "The training is done!" << std::endl;
    // /*
    //     new image recognition
    // */
    // // ---------------------
    // // Recognize New Images
    // // ---------------------
    // /*
    //     initialize the recognition engine
    // */
    // Set device to CPU or CUDA if available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    // Load the model
    std::string modelPath = "/Users/dengfengji/ronnieji/lib/TorchNN/main/siamese_model.pt";
    SiameseNetwork model;
    torch::load(model, modelPath);
    model->to(device);
    model->eval(); // Set model to evaluation mode
    std::vector<std::vector<testImageData>> trainedDataSet;
    std::cout << "Initialize the recognization engine..." << std::endl;
    tnn_j.initialize_embeddings(
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/image.dat",
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/siamese_model.pt",
        "/Users/dengfengji/ronnieji/lib/TorchNN/main/labelMap.txt",
        model,
        device,
        100,//default image size
        trainedDataSet
    );
    /*
        load label mapping file
    */
    std::map<int,std::string> labelMap = tnn_j.read_label_map("/Users/dengfengji/ronnieji/lib/TorchNN/main/labelMap.txt");
    std::cout << "Successfully load the engine, start recognizating..." << std::endl;
    /*
        test input image
        test all image in a folder
    */
    std::vector<std::string> testimgs;
    std::string sub_folder_path = "/Users/dengfengji/ronnieji/Kaggle/imagenet/ILSVRC/Data/CLS-LOC/test"; //"/Users/dengfengji/ronnieji/Kaggle/test";
    for (const auto& entrySubFolder : std::filesystem::directory_iterator(sub_folder_path)) {
        if (!entrySubFolder.is_regular_file()) {
            continue; // Skip non-file entries
        }
        std::string ext = entrySubFolder.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            std::string imgFilePath = entrySubFolder.path().string();
            testimgs.push_back(imgFilePath);
        }
    }
    std::cout << testimgs.size() << std::endl;
    if(trainedDataSet.empty()){
        std::cout << "trainedDataSet is empty!" << std::endl;
        return 0;
    }
    if(!testimgs.empty()){
        for(const auto& item : testimgs){
            auto start = std::chrono::high_resolution_clock::now(); // Initialize start time  
            std::vector<std::string> returnResult;
            tnn_j.testModel(
                item,
                100,
                trainedDataSet,
                labelMap,
                model,
                device,
                returnResult,
                18 //compress image to the same training quality
            );
            if(!returnResult.empty()){
                std::cout << "Image: " << item << std::endl;
                for(const auto& item : returnResult){
                    std::cout << "    has: " << item << std::endl;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();   
            std::chrono::duration<double> duration = end - start;  
            std::cout << "Execution time: " << duration.count() << " seconds\n"; 
        }
    }
    return 0;
}