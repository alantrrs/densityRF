#include <VisualContext.h>

using namespace std;
using namespace cv;

// TRAINING DATA
void ImageCollection::SetData(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& gt, int nclasses,int margin,int step){
  data_ = imgs;
  labels_ = gt;
  nclasses_ = nclasses;
  // Get samples
  for (int i=0;i<data_.size();i++){
    for ( int y=margin+1; y< (data_[i].rows-margin-1); y+=step ){
      for ( int x=margin+1; x<(data_[i].cols-margin-1); x+=step ){ 
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

// STATISTIC AGGREGATOR
double HistogramAggregator::Entropy() const {
  if (sampleCount_ == 0)
    return 0.0;
  double result = 0.0;
  for (int b = 0; b < bins_.size(); b++){
    double p = (double)bins_[b] / (double)sampleCount_;
    result -= p == 0.0 ? 0.0 : p * log(p)/log(2.0);
  }
  return result;
}

HistogramAggregator::HistogramAggregator(){
  sampleCount_ = 0;
}

HistogramAggregator::HistogramAggregator(int nClasses){
  bins_ = vector<unsigned int>(nClasses,0);
  sampleCount_ = 0;
}

float HistogramAggregator::GetProbability(int classIndex) const{
  return (float)(bins_[classIndex]) / sampleCount_;
}

int HistogramAggregator::FindTallestBinIndex() const{
  unsigned int maxCount = bins_[0];
  int tallestBinIndex = 0;
  for (int i = 1; i < bins_.size(); i++){
    if (bins_[i] > maxCount){
      maxCount = bins_[i];
      tallestBinIndex = i;
    }
  }
  return tallestBinIndex;
}

// IStatisticsAggregator implementation
void HistogramAggregator::Clear(){
  for (int b = 0; b < bins_.size(); b++){
    bins_[b] = 0;
  }
  sampleCount_ = 0;
}

void HistogramAggregator::Aggregate(const IDataPointCollection& data, unsigned int index){
  const ImageCollection& concreteData = (const ImageCollection&)(data);
  bins_[concreteData.getLabel((int)index)]+=1;
  sampleCount_ += 1;
}

void HistogramAggregator::Aggregate(const HistogramAggregator& aggregator){
  assert(aggregator.bins_.size() == bins_.size());
  for (int b = 0; b < bins_.size(); b++){
    bins_[b] += aggregator.bins_[b];
  }
  sampleCount_ += aggregator.sampleCount_;
}

HistogramAggregator HistogramAggregator::DeepClone() const{
  HistogramAggregator result(bins_.size());
  for (int b = 0; b < bins_.size(); b++){
    result.bins_[b] = bins_[b];
  }
  result.sampleCount_ = sampleCount_;
  return result;
}
