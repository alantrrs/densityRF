#include <VisualContext.h>
using namespace cv;
using namespace std;

template<class T>
string vec2str(T color){
  stringstream str;
  str << (int)color[0]<< (int)color[1]<<(int)color[2];
  return str.str();
}

Mat colorToLabel(const map<string,int>& labels, const cv::Mat& img){
  Mat tmp;
  cvtColor(img,tmp,CV_BGR2RGB);
  Mat label_img= Mat::zeros(img.rows,img.cols,CV_8U);
  for (int j = 0; j<img.rows;j++){
    for (int i = 0; i<img.cols;i++){
      label_img.at<uchar>(j,i) = (uchar)labels.find(vec2str<Vec3b>(tmp.at<Vec3b>(j,i)))->second; 
    }
  }
  return label_img;
}

void readLabels(FileNode labels_node,map<string,int>& labels,vector<vector<int> >& colors){
  FileNodeIterator lit = labels_node.begin(), lit_end = labels_node.end();
  for (int i=0;lit !=lit_end; lit++,i++){
    vector<int> color;
    (*lit)>> color;
    colors.push_back(color);
    labels[vec2str<vector<int> >(color)]=i;
  }
}

void readImages(FileNode training_files,vector<Mat>& imgs, vector<Mat>& gts){
  FileNodeIterator it = training_files.begin(), it_end = training_files.end();
  for ( ; it != it_end; it++){
    string img_file, gt_file;
    (*it)["img"] >> img_file;
    (*it)["gt"] >> gt_file;
    Mat img, gt, lab_img;
    img = imread(img_file);
    cvtColor(img,lab_img,CV_BGR2Lab);
    gt = imread(gt_file);
    imgs.push_back(lab_img);
    gts.push_back(gt);
  }
}

int main(int argc, char** argv){
  //TODO: 1. Train supervised forest on data to test implementation
  //TODO: 2. Train density forest on data
  //TODO: 3. Using 1 image per class train semi-supervised w fixed features
  //TODO: 4. Train on video
  //Set training parameters
  TrainingParameters parameters;
  parameters.MaxDecisionLevels = 10;
  parameters.NumberOfCandidateFeatures = 50;
  parameters.NumberOfCandidateThresholdsPerFeature = 25;
  parameters.NumberOfTrees = 1;
  parameters.Verbose = 1; 
  //Get Training data
  cout << "Reading data.." << endl;
  FileStorage fs("training.yml",FileStorage::READ);
  //Get Labels
  map<string,int> labels;
  vector<vector<int> > colors;
  readLabels(fs["labels"],labels,colors);
  //Get image files
  vector<Mat> imgs, gts,color_images;
  FileNode training_files = fs["files"];
  readImages(training_files,imgs,color_images);
  //Get label images (conver color to label)
  for (int i=0;i<color_images.size();i++){
    Mat label_img = colorToLabel(labels,color_images[i]);
    gts.push_back(label_img);
  }
  //Set data
  int margin = 20;
  ImageCollection trainingData;
  trainingData.SetData(imgs,gts,(int)labels.size(),margin);
  //trainingData.show();
  cout << "Training samples: " << trainingData.Count() << endl;
  //Train Forest
  Random random;
  ClassificationContext<PixelCompFeature> context(trainingData);
  std::auto_ptr<Forest<PixelCompFeature,HistogramAggregator> > forest = ForestTrainer<PixelCompFeature,HistogramAggregator>::TrainForest(random,parameters,context,trainingData);
  //Load test data
  FileStorage tfs("test.yml",FileStorage::READ);
  ImageCollection testData;
  //Get test image files
  color_images.clear();
  imgs.clear();
  gts.clear();
  readImages(tfs["files"],imgs,color_images);
  //Get label images (conver color to label)
  for (int i=0;i<color_images.size();i++){
    Mat label_img = colorToLabel(labels,color_images[i]);
    gts.push_back(label_img);
  }
  int nClasses = forest->GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();
  testData.SetData(imgs,gts,nClasses,margin,1);
  // Test Forest
  std::vector<std::vector<int> > leafIndicesPerTree;
  forest->Apply(testData, leafIndicesPerTree);
  HistogramAggregator posteriors;
  // Initialize images;
  vector<Mat> out_imgs(imgs.size());
  for (int i = 0; i<imgs.size();i++){
   out_imgs[i]=Mat::zeros(imgs[i].rows,imgs[i].cols,CV_8UC3);
  }
  vector<int> total_samples(nClasses,0);
  vector<int> correct_samples(nClasses,0);
  for (int i = 0; i < testData.Count(); i++){
    // Get Random Forest posteriors
    posteriors = HistogramAggregator(nClasses);
    for (int t = 0; t < forest->TreeCount(); t++){
      int leafIndex = leafIndicesPerTree[t][i];
      posteriors.Aggregate(forest->GetTree(t).GetNode(leafIndex).TrainingDataStatistics);
    }
    int c =  posteriors.FindTallestBinIndex();
    //Evaluate accuracy
    int gt_label = testData.getLabel(i);
    total_samples[gt_label]+=1;
    if (c == gt_label)
      correct_samples[gt_label]+=1;
    //Change to color
    Vec3b color(colors[c][2],colors[c][1],colors[c][0]);
    DataPoint p = testData.getDataPoint(i); 
    out_imgs[p.i].at<Vec3b>(p.pt) =color;
  }
  // Evaluate
  for (int c=0; c<total_samples.size();c++){
    printf("Class: %d Samples %d Accuracy %f%%\n",c,total_samples[c],(total_samples[c]>0)? 100.0*correct_samples[c]/(float)total_samples[c] : 0); 
  }
  //Save images
  for (int i=0;i<out_imgs.size();i++){
    stringstream ss;
    ss << "img_" << i << ".bmp"; 
    imwrite(ss.str(),out_imgs[i]);
  }
}
