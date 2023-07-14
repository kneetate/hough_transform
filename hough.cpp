#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>

struct Line
{
    double rho;
    double theta;
};

class HoughLineDetector
{
    double _rho_step = 1;
    double _theta_step = 1;
    double _theta_max = 90;
    double _theta_min = 0;
    int _vote_thresh = 150;
    cv::Mat _vote_map;

public:
    void setParam(double rho_step, double theta_step, double theta_max, double theta_min)
    {
        _rho_step = rho_step;
        _theta_step = theta_step;
        _theta_max = theta_max;
        _theta_min = theta_min;
    }
    void detect(cv::Mat &mat, cv::Mat &mask, int thresh = 10)
    {
        _vote_thresh = thresh;
        _vote_map = cv::Mat(cv::Size(int((_theta_max - _theta_min) / _theta_step), int(std::max(mat.cols, mat.rows) / _rho_step)), CV_32S, cv::Scalar(0));
        std::cout << _vote_map.size() << std::endl;
        std::cout << mat.size() << std::endl;
        // mat.ptr<unsigned char>(706)[586];
        // _vote_map.ptr<int>(716)[89] += 1;
        int rho = 0, theta = 0;
        for (size_t j = 0; j < mat.rows; j++)
        {
            for (size_t i = 0; i < mat.cols; i++)
            {
                if (mat.ptr<unsigned char>(j)[i] > 0)
                {
                    for (double t = 0; t < _theta_max - _theta_min; t += _theta_step)
                    {
                        rho = int((double(i) * cos(t * M_PI / 180) + double(j) * sin(t * M_PI / 180)) / _rho_step);
                        theta = int(t);
                        if ((rho < _vote_map.rows && rho >= 0) && (theta < _vote_map.cols && theta >= 0))
                        {
                            _vote_map.ptr<int>(rho)[theta] += 1;
                        }
                    }
                }
            }
        }
        std::cout << "finish loop" << std::endl;
        std::vector<Line> detected_line;
        for (size_t j = 0; j < _vote_map.rows; j++)
        {
            for (size_t i = 0; i < _vote_map.cols; i++)
            {
                if (_vote_map.ptr<int>(j)[i] > _vote_thresh)
                {
                    detected_line.push_back({j * _rho_step, i * _theta_step});
                }
            }
        }
        mask = cv::Mat(mat.size(), CV_8U, cv::Scalar(0));
        for (auto &l : detected_line)
        {
            double a = -cos(l.theta * M_PI / 180) / sin(l.theta * M_PI / 180);
            double b = l.rho / sin(l.theta * M_PI / 180);
            for (size_t i = 0; i < mat.cols; i++)
            {
                int j = int(a * i + b);
                if ((i < mask.cols && i >= 0) && (j < mask.rows && j >= 0))
                    mask.ptr<unsigned char>(j)[i] = 255;
            }
        }

        std::cout << "detected " << detected_line.size() << std::endl;
    }
};

struct Curve
{
    Curve() {}
    Curve(double a_, double p_, double q_)
    {
        a = a_;
        p = p_;
        q = q_;
    }
    double a;
    double p;
    double q;
};

class HoughCurveDetector
{
    double _a_step = 0.005;
    double _p_step = 1;
    double _q_step = 1;
    double _a_max = 1.0;
    double _a_min = -1.0;
    double _p_max = 150.0;
    double _p_min = -150.0;
    double _q_max = 0;
    double _q_min = -150;
    int _vote_thresh = 60;
    cv::Mat _vote_map;

public:
    void detect(cv::Mat &mat, cv::Mat &mask, int thresh = 10)
    {
        _vote_thresh = thresh;
        const int a_depth = int((_a_max - _a_min) / _a_step);
        const int p_depth = int((_p_max - _p_min) / _p_step);
        const int q_depth = int((_q_max - _q_min) / _q_step);
        const int sizes[3] = {a_depth, p_depth, q_depth};
        _vote_map = cv::Mat(3, sizes, CV_32S);
        int rho = 0, theta = 0;
        for (size_t j = 0; j < mat.rows; j++)
        {
            for (size_t i = 0; i < mat.cols; i++)
            {
                if (mat.ptr<unsigned char>(j)[i] > 0)
                {
                    double d_a = 0, d_p = 0, d_q = 0;
                    int q = 0;
                    for (size_t a = 0; a < a_depth; a++)
                    {
                        for (size_t p = 0; p < p_depth; p++)
                        {
                            d_a = _a_min + a * _a_step;
                            d_p = _p_min + p * _p_step;
                            d_q = j - d_a * (i - d_p) * (i - d_p);
                            q = int((d_q - _q_min) / _q_step);
                            if (a < a_depth && a >= 0 && p < p_depth && p >= 0 && q < q_depth && q >= 0)
                            {
                                _vote_map.at<int>(a, p, q) += 1;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "finish loop" << std::endl;
        std::vector<Curve> detected_curve;
        for (size_t q = 0; q < q_depth; q++)
        {
            for (size_t p = 0; p < p_depth; p++)
            {
                for (size_t a = 0; a < a_depth; a++)
                {
                    if (_vote_map.at<int>(a, p, q) > _vote_thresh)
                    {
                        detected_curve.push_back({double(a), double(p), double(q)});
                    }
                }
            }
        }
        std::cout << "detected " << detected_curve.size() << std::endl;
        mask = cv::Mat(mat.size(), CV_8U, cv::Scalar(0));
        for (auto &c : detected_curve)
        {
            double a = _a_min + c.a * _a_step;
            double p = _p_min + c.p * _p_step;
            double q = _q_min + c.q * _q_step;
            for (size_t i = 0; i < mat.cols; i++)
            {
                int j = int(a * (i - p) * (i - p) + q);
                if ((i < mask.cols && i >= 0) && (j < mask.rows && j >= 0))
                {
                    mask.ptr<unsigned char>(j)[i] = 255;
                }
            }
        }
    }
};

int main(int argc, char **argv)
{
    cv::Mat mat(cv::Size(500, 500), CV_8U, cv::Scalar(0));
    cv::Mat raw(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
    // curve paramters for 4 curve
    double a[4] = {0.05, 0.01, 0.02, 0.03};
    double p[4] = {-30.0, -100.0, 50.0, 120.0};
    double q[4] = {-40.0, -30, -80, -130};

    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());

    std::normal_distribution<> dist(0.0, 5.0);

    // drawing 4 curves
    for (size_t i = 0; i < mat.cols; i++)
    {
        for (int k = 0; k < 4; k++)
        {
            int j = int(a[k] * (i - p[k]) * (i - p[k]) + q[k]) + dist(engine);
            if ((i < mat.cols && i >= 0) && (j < mat.rows && j >= 0))
            {
                mat.ptr<unsigned char>(j)[i] = 255;
                raw.ptr<cv::Vec3b>(j)[i] = cv::Vec3b(255, 255, 255);
            }
        }
    }
    HoughCurveDetector curve;
    cv::Mat mask;
    int thresh = 10;
    curve.detect(mat, mask, 10);

    // overwrite the detected curves on the raw mat
    for (size_t j = 0; j < mat.rows; j++)
    {
        for (size_t i = 0; i < mat.cols; i++)
        {
            if (mask.ptr<unsigned char>(j)[i] > 0)
                raw.ptr<cv::Vec3b>(j)[i] = cv::Vec3b(0, 0, 255);
        }
    }
    cv::imshow("result", raw);
    cv::waitKey(0);
    return 0;
}