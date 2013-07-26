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
  int margin_;
  int nclasses_;
  public:
  void SetData(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat>& gt, int nclasses,int margin,int step=2);
  void show();
  unsigned int Count() const;
  int CountClasses() const;
  DataPoint getDataPoint(int i) const;
  int getLabel(int i) const;
  int getMargin() const { return margin_;}
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

