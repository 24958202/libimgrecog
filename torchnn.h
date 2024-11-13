#ifndef TORCHNN_H
#define TORCHNN_H
#include "project_includes.h"
#include "subfunctions.cpp"
#ifdef __cplusplus
extern "C" {
#endif
class torchnn{
    public:
        /*
            On Linux, you might use a package manager like apt:
                sudo apt install libjpeg-turbo8-dev
            On macOS, you can use Homebrew:
                brew install jpeg

            Function to compress an image
            para1: input image path
            para2: output image path
            para3: quality (much be the same as training quality)
        */
        void compressJPEG(const std::string&, const std::string&, int);
        /*
            Function to compress all images in the training folder:
            para1: input folder path
                    training folder:
                    main folder
                    |------------
                    |       |
                    catalog1, catalog2 ....
            para2: quality (much be the same as training quality)
        */
        void img_compress(const std::string&, int);
        /*
            serialize all images into an image.dat file
            para1: imageSize(default 100)->(width:100pixel, height: 100pixel)
            para2: mainFolderPath corpus image folder path
                    training folder:
                    main folder
                    |------------
                    |       |
                    catalog1, catalog2 ....
            para3: output image.dat path
        */
        void serializeImages(int, const std::string&, const std::string&) const;
        /*
            load image.dat and initialize labels of the image
            para1: image.dat file path
            para2: image size(default 100)->(width:100pixel, height: 100pixel)
            para3: label mapping file (labelMap.txt file to store label and catalogs mapping)
        */
        std::vector<ImageData> loadDataset(const std::string&, int, const std::string&);
        /*
            train image recognition model
            para1: image.dat dataset path
            para2: output model path siamese_model.pt,
            para3: output label mapping file path labelMap.txt\
            para4: image size (default 100) ->(width:100pixel, height: 100pixel)
            para5: numEpochs (default: 10)
            para6: batchSize (default: batchSize)
            para7: learningRate (default: 1e-3)
        */
        void train_model(const std::string&,const std::string&,const std::string&,int,int,int,float);
        /*
            Function to load label mapping file labelMap.txt into dataset std::map<int,std::string>
            para1: input labelMap.txt file path
        */
        std::map<int,std::string> read_label_map(const std::string&);
        /*
            initialize image model, prepare for image recognition
            para1: image.dat file path
            para2: model path siamese_model.dt
            para3: input SiameseNetwork model
            para4: input torch::Device device
            para5: image size (default 100) ->(width:100pixel, height: 100pixel)
            para6: output trained dataset for descriptors comparing
        */
        void initialize_embeddings(
            const std::string&, 
            const std::string&, 
            const std::string&, 
            SiameseNetwork&, 
            torch::Device&, 
            int, 
            std::vector<std::vector<testImageData>>&
        );
        /*
            Function to recognition a new image
            para1: input image path
            para2: imageSize (default 100) ->(width:100pixel, height: 100pixel)
            para3: trained dataset from initialize_embeddings
            para4: image label mapping file path labelMap.txt
            para5: input model from initialize_embeddings
            para6: input device from initialize_embeddings
            para7: output results
            para8, int, image quality (much be the same as training quality, default: 18) 
        */
        void testModel(const std::string&, int, const std::vector<std::vector<testImageData>>&, const std::map<int, std::string>&, SiameseNetwork&, torch::Device&, std::vector<std::string>&, int);

};
#ifdef __cplusplus
}
#endif

#endif