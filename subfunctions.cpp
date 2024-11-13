#ifndef SUBFUNCTIONS_CPP
#define SUBFUNCTIONS_CPP
#include "project_includes.h"
// ---------------------
// Data Preparation
// ---------------------
struct ImageData {
    torch::Tensor imageTensor;
    int label;
};
struct testImageData {
    torch::Tensor imageTensor;
    torch::Tensor embedding;
    int label;
};
inline void removeDuplicates(std::vector<std::string>& vec) {
    if(vec.empty()){
        return;
    }
    // Step 1: Sort the vector
    std::sort(vec.begin(), vec.end());
    // Step 2: Use std::unique to remove duplicates
    auto last = std::unique(vec.begin(), vec.end());
    // Step 3: Resize the vector to eliminate the undefined part
    vec.erase(last, vec.end());
}
inline bool isValidImageFile(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png");
}
inline std::vector<std::string> JsplitString(const std::string& input, char delimiter){
    std::vector<std::string> result;
    if(input.empty() || delimiter == '\0'){
        std::cerr << "Jsonlib::splitString input empty" << '\n';
    	return result;
    }
    std::stringstream ss(input);
    std::string token;
    while(std::getline(ss,token,delimiter)){
        result.push_back(token);
    }
    return result;
}
inline void compressJPEG(const std::string& inputFilename, const std::string& outputFilename, int quality) {
    if(inputFilename.empty() || outputFilename.empty()){
        return;
    }
    // Create a JPEG compression struct and error handler
    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    // Open the input file
    FILE* infile = fopen(inputFilename.c_str(), "rb");
    if (!infile) {
        throw std::runtime_error("Unable to open input file: " + inputFilename);
    }
    // Initialize JPEG decompression
    jpeg_decompress_struct dinfo;
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);
    jpeg_stdio_src(&dinfo, infile);
    jpeg_read_header(&dinfo, TRUE);
    jpeg_start_decompress(&dinfo);
    // Get image properties
    int width = dinfo.output_width;
    int height = dinfo.output_height;
    int numChannels = dinfo.num_components;
    // Allocate memory for the image data
    unsigned char* buffer = new unsigned char[width * height * numChannels];
    while (dinfo.output_scanline < height) {
        unsigned char* row_pointer = buffer + dinfo.output_scanline * width * numChannels;
        jpeg_read_scanlines(&dinfo, &row_pointer, 1);
    }
    // Finish decompression and close the input file
    jpeg_finish_decompress(&dinfo);
    fclose(infile);
    jpeg_destroy_decompress(&dinfo);
    // Set up compression
    FILE* outfile = fopen(outputFilename.c_str(), "wb");
    if (!outfile) {
        delete[] buffer;
        throw std::runtime_error("Unable to open output file: " + outputFilename);
    }
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = numChannels;
    cinfo.in_color_space = (numChannels == 3) ? JCS_RGB : JCS_GRAYSCALE;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_stdio_dest(&cinfo, outfile);
    // Start compression
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char* row_pointer = buffer + cinfo.next_scanline * width * numChannels;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    // Finish compression and clean up
    jpeg_finish_compress(&cinfo);
    fclose(outfile);
    jpeg_destroy_compress(&cinfo);
    delete[] buffer;
    std::cout << "JPEG image compressed and saved to " << outputFilename << std::endl;
}
inline std::map<int,std::string> read_label_map(const std::string& lable_map_path){
    std::map<int,std::string> label_mapping;
    if(lable_map_path.empty()){
        return label_mapping;
    }
    std::ifstream ifile(lable_map_path);
    if(ifile.is_open()){
        std::string line;
        while(std::getline(ifile, line)){
            if(!line.empty()){
                std::vector<std::string> get_line = JsplitString(line,',');
                if(!get_line.empty()){
                    int label = std::stoi(get_line[0]);
                    std::string className = get_line[1];
                    label_mapping[label] = className;
                }
            }
        }
    }
    ifile.close();
    return label_mapping;
}
// Function to load images from a .dat file
// Function to load images from a .dat file
inline std::map<std::string, std::vector<cv::Mat>> loadImages(int imageSize,const std::string& inputPath) {
    std::ifstream inFile(inputPath, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error opening input file for reading: " << inputPath << std::endl;
        return {};
    }
    std::map<std::string, std::vector<cv::Mat>> dataset;
    while (true) {
        // Read catalog name length
        size_t catalogLength;
        if (!inFile.read(reinterpret_cast<char*>(&catalogLength), sizeof(size_t))) break;
        // Read catalog name
        std::string catalog(catalogLength, '\0');
        inFile.read(&catalog[0], catalogLength);
        // Read image size
        size_t serializedImageSize;
        inFile.read(reinterpret_cast<char*>(&serializedImageSize), sizeof(size_t));
        // Read image data
        std::vector<uchar> buffer(serializedImageSize);
        inFile.read(reinterpret_cast<char*>(buffer.data()), serializedImageSize);
        // Create cv::Mat for the image directly from buffer
        cv::Mat image(imageSize, imageSize, CV_32FC3, buffer.data());  // Assuming images are resized to (imageSize, imageSize)
        if (!image.empty()) {
            // No need to convert back to CV_32F, as it's already in the tensor format
            dataset[catalog].push_back(image.clone());  // Clone to store in the map
        } else {
            std::cerr << "Failed to create image for catalog: " << catalog << std::endl;
        }
    }
    inFile.close();
    return dataset;
}
/*
    end Serialize the image
*/
// ---------------------
// Siamese Network Model
// ---------------------
struct SubNetImpl : torch::nn::Module {
    torch::nn::Sequential features;
    SubNetImpl() {
        features = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(1)
        );
        register_module("features", features);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = x.view({x.size(0), -1}); // Flatten
        return x;
    }
};
TORCH_MODULE(SubNet);
struct SiameseNetworkImpl : torch::nn::Module {
    SubNet subnetwork;
    SiameseNetworkImpl() {
        subnetwork = SubNet();
        register_module("subnetwork", subnetwork);
    }
    torch::Tensor forward(torch::Tensor input1, torch::Tensor input2) {
        auto output1 = subnetwork->forward(input1);
        auto output2 = subnetwork->forward(input2);
        // Compute Euclidean distance between embeddings
        auto distance = torch::pairwise_distance(output1, output2, 2); // p=2 for Euclidean distance
        return distance;
    }
};
TORCH_MODULE(SiameseNetwork);
// ---------------------
// Contrastive Loss Function
// ---------------------
inline torch::Tensor contrastiveLoss(torch::Tensor distance, torch::Tensor label, float margin = 1.0f) {
    // label: 1 for similar pairs, 0 for dissimilar pairs
    auto loss = label * torch::pow(distance, 2) +
                (1 - label) * torch::pow(torch::clamp(margin - distance, /*min=*/0.0f), 2);
    return loss.mean();
}
/*
    load
*/
inline torch::Tensor load_and_preprocess_image(const std::string& imagePath, int imageSize, int quality) {
    if(imagePath.empty()){
        return torch::Tensor();
    }
    /*
        compress the image to the quality
    */
    compressJPEG(imagePath,imagePath,quality);
    /*
        end compressing
    */
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Could not read image: " << imagePath << std::endl;
        return torch::Tensor();
    }
    // Resize and convert to RGB
    cv::resize(img, img, cv::Size(imageSize, imageSize));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // Convert to float tensor and normalize
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat);
    tensor = tensor.permute({2, 0, 1}); // Change to C x H x W
    // Normalize using mean and std of ImageNet dataset
    tensor = tensor.clone(); // Ensure that tensor owns its memory
    tensor[0] = tensor[0].sub_(0.485).div_(0.229);
    tensor[1] = tensor[1].sub_(0.456).div_(0.224);
    tensor[2] = tensor[2].sub_(0.406).div_(0.225);
    return tensor;
}
inline void createImagePairs(const std::vector<ImageData>& dataset,
                      std::vector<std::pair<torch::Tensor, torch::Tensor>>& imagePairs,
                      std::vector<int>& pairLabels) {
    // Create similar and dissimilar pairs
    std::vector<ImageData> shuffledDataset = dataset;
    std::shuffle(shuffledDataset.begin(), shuffledDataset.end(), std::mt19937{std::random_device{}()});
    for (size_t i = 0; i < shuffledDataset.size(); ++i) {
        // Positive pair
        for (size_t j = i + 1; j < shuffledDataset.size(); ++j) {
            if (shuffledDataset[i].label == shuffledDataset[j].label) {
                imagePairs.push_back({shuffledDataset[i].imageTensor, shuffledDataset[j].imageTensor});
                pairLabels.push_back(1);
                break; // Limit positive pairs
            }
        }
        // Negative pair
        for (size_t j = i + 1; j < shuffledDataset.size(); ++j) {
            if (shuffledDataset[i].label != shuffledDataset[j].label) {
                imagePairs.push_back({shuffledDataset[i].imageTensor, shuffledDataset[j].imageTensor});
                pairLabels.push_back(0);
                break; // Limit negative pairs
            }
        }
    }
    // Shuffle pairs
    std::shuffle(imagePairs.begin(), imagePairs.end(), std::mt19937{std::random_device{}()});
}
// ---------------------
// Training Function
// ---------------------
inline void trainModel(SiameseNetwork& model,
                std::vector<std::pair<torch::Tensor, torch::Tensor>>& imagePairs,
                std::vector<int>& pairLabels,
                torch::optim::Adam& optimizer,
                int epochs = 10,
                int batch_size = 16) {
    model->train();
    size_t dataset_size = imagePairs.size();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;
        int batches = 0;
        for (size_t i = 0; i < dataset_size; i += batch_size) {
            size_t end = std::min(i + batch_size, dataset_size);
            std::vector<torch::Tensor> batch_input1, batch_input2, batch_labels;
            for (size_t j = i; j < end; ++j) {
                batch_input1.push_back(imagePairs[j].first);
                batch_input2.push_back(imagePairs[j].second);
                batch_labels.push_back(torch::tensor(static_cast<float>(pairLabels[j])));
            }
            auto input1 = torch::stack(batch_input1).to(torch::kFloat32);
            auto input2 = torch::stack(batch_input2).to(torch::kFloat32);
            auto labels = torch::stack(batch_labels);
            optimizer.zero_grad();
            auto distances = model->forward(input1, input2);
            auto loss = contrastiveLoss(distances, labels);
            loss.backward();
            optimizer.step();
            epoch_loss += loss.item<double>();
            ++batches;
        }
        std::cout << "Epoch [" << epoch << "/" << epochs << "] Loss: " << epoch_loss / batches << std::endl;
    }
}
// ---------------------
// Evaluation Function
// ---------------------
inline float computeSimilarity(SiameseNetwork& model, torch::Tensor image1, torch::Tensor image2) {
    model->eval(); // Set the model to evaluation mode
    torch::NoGradGuard no_grad; // Disable gradient computation
    image1 = image1.unsqueeze(0); // Add batch dimension
    image2 = image2.unsqueeze(0);
    auto distance = model->forward(image1, image2);
    return distance.item<float>(); // Convert to scalar
}
inline double computeCosineSimilarity(const torch::Tensor& tensor1, const torch::Tensor& tensor2) {
    // Use torch::Tensor as input and specify the dimension using the options struct
    auto similarity = torch::nn::functional::cosine_similarity(tensor1, tensor2, torch::nn::CosineSimilarityOptions().dim(0));
    return similarity.item<double>();
}
inline int findMostSimilarImageLabel(const torch::Tensor& newImageEmbedding, const std::vector<testImageData>& dataset) {
    double highestSimilarity = -1.0;
    int mostSimilarLabel = -1;
    for (const auto& data : dataset) {
        double similarity = computeCosineSimilarity(newImageEmbedding, data.embedding);
        if (similarity > highestSimilarity) {
            highestSimilarity = similarity;
            mostSimilarLabel = data.label;
        }
    }
    return mostSimilarLabel;
}
inline void computeAndSaveEmbeddings(const std::string& dataset_dat_Path, SiameseNetwork& model, torch::Device& device, const std::string& embeddingsPath, int imageSize) {
    if(dataset_dat_Path.empty()){
        std::cerr << "computeAndSaveEmbeddings: dataset_dat_Path or labelMap_path is empty!" << std::endl;
        return;
    }
    std::map<std::string, std::vector<cv::Mat>> img_to_train = loadImages(imageSize,dataset_dat_Path);
    if(!img_to_train.empty()){
        for(const auto& item : img_to_train){
            std::cout << "Processing class: " << item.first << std::endl;
            const auto& item_value = item.second;
            unsigned int img_count = 0;
            for(const auto& img : item_value){
                img_count++;
                auto imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
                imgTensor = imgTensor.permute({2, 0, 1}); // [H,W,C] -> [C,H,W]
                imgTensor = imgTensor.clone(); // Ensure that tensor owns its memory
                imgTensor[0] = imgTensor[0].sub_(0.485).div_(0.229);
                imgTensor[1] = imgTensor[1].sub_(0.456).div_(0.224);
                imgTensor[2] = imgTensor[2].sub_(0.406).div_(0.225);
                if (imgTensor.numel() == 0) continue;
                // Move to device
                imgTensor = imgTensor.to(device);
                // Compute embedding
                torch::Tensor embedding = model->subnetwork->forward(imgTensor);
                embedding = embedding.flatten(); // Flatten embedding
                std::string ofile_name = item.first;
                ofile_name.append(std::to_string(img_count));
                // Save the embedding (std::filesystem::path(imagePath).stem().string())
                if(!std::filesystem::exists(embeddingsPath)){
                    if(!std::filesystem::create_directory(embeddingsPath)){
                        std::cout << "Failed to create the directory: " << embeddingsPath << std::endl;
                        return;
                    }
                }
                std::string embeddingFilename = embeddingsPath + "/" + ofile_name + ".pt";
                torch::save(embedding, embeddingFilename);
            }
        }
    }
    else{
        std::cerr << "computeAndSaveEmbeddings img_to_train is empty!" << std::endl;
    }
}
inline std::vector<std::vector<testImageData>> loadDatasetWithEmbeddings(const std::string& dataset_dat_Path, const std::string& labelMap_path, int imageSize, SiameseNetwork& model, torch::Device& device) {
    std::vector<std::vector<testImageData>> dataset;
    if (dataset_dat_Path.empty()) {
        std::cerr << "Dataset image.dat file path is empty!" << std::endl;
        return dataset;
    }
    // Load label mapping file
    std::map<int, std::string> label_m = read_label_map(labelMap_path);
    if (label_m.empty()) {
        std::cerr << "Label map file is empty or not found!" << std::endl;
        return dataset;
    }
    std::map<std::string, std::vector<cv::Mat>> img_to_train = loadImages(imageSize,dataset_dat_Path);
    if(!img_to_train.empty()){
        for(const auto& item : img_to_train){
            std::vector<testImageData> d_item;
            int currentLabel = -1;
            for (const auto& lbm : label_m) {
                if (lbm.second == item.first) {
                    currentLabel = lbm.first;
                    break;
                }
            }
            if (currentLabel == -1) {
                std::cerr << "No matching label found for directory: " << item.first << std::endl;
                continue;
            }
            const auto& item_value = item.second;
            for(const auto& img : item_value){
                auto imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32);
                imgTensor = imgTensor.permute({2, 0, 1}); // [H,W,C] -> [C,H,W]
                imgTensor = imgTensor.clone(); // Ensure that tensor owns its memory
                imgTensor[0] = imgTensor[0].sub_(0.485).div_(0.229);
                imgTensor[1] = imgTensor[1].sub_(0.456).div_(0.224);
                imgTensor[2] = imgTensor[2].sub_(0.406).div_(0.225);
                if (imgTensor.numel() > 0) {
                    imgTensor = imgTensor.to(device);
                    torch::Tensor embedding = model->subnetwork->forward(imgTensor);
                    embedding = embedding.flatten();
                    d_item.push_back({imgTensor, embedding, currentLabel});
                } else {
                    std::cerr << "Failed to load one of the catalog preprocess image: " << item.first << std::endl;
                }
            }
            if (!d_item.empty()) {
                dataset.push_back(d_item);
            }
        }
    }
    return dataset;
}
#endif