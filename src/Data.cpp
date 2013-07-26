#include <Data.h>

using namespace std;
using namespace cv;

// TRAINING DATA
void ImageCollection::SetData(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& gt, int nclasses,int margin,int step){
  margin_ =margin;
  data_ = imgs;
  labels_ = gt;
  nclasses_ = nclasses;
  // Get samples
  for (int i=0;i<data_.size();i++){
    for ( int y=margin_+1; y< (data_[i].rows-margin_-1); y+=step ){
      for ( int x=margin_+1; x<(data_[i].cols-margin_-1); x+=step ){ 
        DataPoint p;
        p.pt = Point(x,y);
        p.i = i;
        positions_.push_back(p);
      }
    }
  }
  if (labels_.size()==0){
   printf("No ground truth data assigned\n");
  }
}
/// Count the data points in this collection.
unsigned int ImageCollection::Count() const{
  return positions_.size();
}

int ImageCollection::CountClasses() const{
  return nclasses_;
}

int ImageCollection::getLabel(int i) const{
  DataPoint p = positions_[i];
  return (int)labels_[p.i].at<uchar>(p.pt);
}   

DataPoint ImageCollection::getDataPoint(int i) const{
  return positions_[i];
}

void ImageCollection::show(){
  for (int i=0; i<data_.size();i++){
    cv::imshow("img",data_[i]);
    cv::imshow("gt",labels_[i]);
    char k = cv::waitKey(30);
    if (k=='p')
      cv::waitKey(0);
  }
}

void ImageCollection::release(){
  data_.clear();
  labels_.clear();
  positions_.clear();
}

