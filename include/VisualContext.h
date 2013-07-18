#include <Sherwood.h>
#include <opencv2/opencv.hpp>

using namespace MicrosoftResearch::Cambridge::Sherwood;

struct DataPoint {
  int i;
  cv::Point2d pt;
};

class ImageCollection : public IDataPointCollection {
  std::vector<DataPoint> positions_;
  std::vector<cv::Mat> data_;
  std::vector<cv::Mat> labels_;
  int nclasses_;
  public:
  void SetData(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& gt, int nclasses,int margin,int step=2);
  void show();
  unsigned int Count() const;
  int CountClasses() const;
  DataPoint getDataPoint(int i) const;
  int getLabel(int i) const;
  void release();
friend class PixelCompFeature;
};

class PixelCompFeature {
  cv::Point2d off1_;
  cv::Point2d off2_;
  int chan_;
  public:
  PixelCompFeature(){
   chan_=-1;
  }
  PixelCompFeature(cv::Point2d off1, cv::Point2d off2,int chan):
    off1_(off1), off2_(off2), chan_(chan){}
  float GetResponse(const IDataPointCollection& data, unsigned int dataIndex)const {
    const ImageCollection& cdata = (ImageCollection&)(data);
    DataPoint dp = cdata.positions_[dataIndex];
    float f = abs(cdata.data_[dp.i].at<cv::Vec3b>(dp.pt+off1_)[chan_]-cdata.data_[dp.i].at<cv::Vec3b>(dp.pt+off2_)[chan_]); 
    return f;
  } 
};

template<class F>
class FeatureGenerator{
  public:
  virtual F createRandom(Random& rand,int maxOffset)=0;
};
class PixelCompGenerator : public FeatureGenerator<PixelCompFeature>{
  public:
  PixelCompFeature createRandom(Random& random,int maxOffset){
    cv::Point p1(random.Next(0,maxOffset),random.Next(0,maxOffset));
    cv::Point p2(random.Next(0,maxOffset),random.Next(0,maxOffset));
    int channel = random.Next(0,3);
    return PixelCompFeature(p1,p2,channel);
  }
};

class HistogramAggregator{
    std::vector<unsigned int> bins_;
    unsigned int sampleCount_;
  public:
    double Entropy() const;
    HistogramAggregator();
    HistogramAggregator(int nClasses);
    float GetProbability(int classIndex) const;
    int BinCount(){ return bins_.size();}
    unsigned int SampleCount() const { return sampleCount_; }
    int FindTallestBinIndex() const;
    // IStatisticsAggregator implementation
    void Clear();
    void Aggregate(const IDataPointCollection& data, unsigned int index);
    void Aggregate(const HistogramAggregator& aggregator);
    HistogramAggregator DeepClone() const;
};

template<class F>
class ClassificationContext : public ITrainingContext<F,HistogramAggregator> {
  private:
    int nClasses_;
    int maxOffset_;
  public:
    ClassificationContext(int nClasses, int maxOffset) {
      nClasses_ = nClasses;
      maxOffset_ = maxOffset;
    }
  private:
    // Implementation of ITrainingContext
    F GetRandomFeature(Random& random){
    cv::Point p1(random.Next(0,maxOffset_),random.Next(0,maxOffset_));
    cv::Point p2(random.Next(0,maxOffset_),random.Next(0,maxOffset_));
    int channel = random.Next(0,3);
    return PixelCompFeature(p1,p2,channel);
 
   }

    HistogramAggregator GetStatisticsAggregator(){
      return HistogramAggregator(nClasses_);
    }
    double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics){
      double entropyBefore = allStatistics.Entropy();
      unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
      if (nTotalSamples <= 1)
        return 0.0;
      double le = leftStatistics.Entropy();
      double re = rightStatistics.Entropy();
      double entropyAfter = (leftStatistics.SampleCount() * le + rightStatistics.SampleCount() * re) / nTotalSamples;
     return entropyBefore - entropyAfter;
    }
    bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain){
      return gain < 0.01;
    }
};
