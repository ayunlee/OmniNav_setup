#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "mm/Frame.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

class OccupancyGridMap {
public:
    OccupancyGridMap(int num_rows, int num_cols, double meters_per_cell, const std::vector<double>& grid_origin_in_map_frame)
    : num_rows_(num_rows), num_cols_(num_cols), meters_per_cell_(meters_per_cell), init_log_odds_(0) {
        log_odds_ratio_occupancy_grid_map_ = std::vector<std::vector<double>>(num_rows_, std::vector<double>(num_cols_, init_log_odds_));
        map_info_.resolution = meters_per_cell_;
        map_info_.width = num_cols_;
        map_info_.height = num_rows_;
        map_info_.origin.position.x = grid_origin_in_map_frame[0];
        map_info_.origin.position.y = grid_origin_in_map_frame[1];
        map_info_.origin.position.z = grid_origin_in_map_frame[2];
        map_info_.origin.orientation.x = 0;
        map_info_.origin.orientation.y = 0;
        map_info_.origin.orientation.z = 0;
        map_info_.origin.orientation.w = 1;
        seq_ = 0;
    }

    void updateLogOddsRatio(int row, int col, double delta_log_odds) {
        if (row >= 0 && row < num_rows_ && col >= 0 && col < num_cols_) {
            log_odds_ratio_occupancy_grid_map_[row][col] = std::clamp(log_odds_ratio_occupancy_grid_map_[row][col] + delta_log_odds, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
        }
    }

    std::pair<int, int> cartesianToGridCoords(double x, double y) {
        int col = static_cast<int>((x - map_info_.origin.position.x) / map_info_.resolution);
        int row = static_cast<int>((y - map_info_.origin.position.y) / map_info_.resolution);
        return {row, col};
    }

    void grow(int min_row, int min_col, int max_row, int max_col) {
        int padding = 50;  // 최소 여유분을 추가하기 위한 패딩 값
        int pad_top = (min_row < 0) ? -min_row + padding : 0;
        int pad_bottom = (max_row >= num_rows_) ? (max_row - num_rows_ + 1) + padding : 0;
        int pad_left = (min_col < 0) ? -min_col + padding : 0;
        int pad_right = (max_col >= num_cols_) ? (max_col - num_cols_ + 1) + padding : 0;

        int new_num_rows = num_rows_ + pad_top + pad_bottom;
        int new_num_cols = num_cols_ + pad_left + pad_right;

        ROS_INFO("expand (%d, %d) => (%d, %d)", num_rows_, num_cols_, new_num_rows, new_num_cols);

        std::vector<std::vector<double>> new_map(new_num_rows, std::vector<double>(new_num_cols, init_log_odds_));

        for (int r = 0; r < num_rows_; ++r) {
            for (int c = 0; c < num_cols_; ++c) {
                new_map[r + pad_top][c + pad_left] = log_odds_ratio_occupancy_grid_map_[r][c];
            }
        }

        log_odds_ratio_occupancy_grid_map_ = new_map;
        num_rows_ = new_num_rows;
        num_cols_ = new_num_cols;
        map_info_.width = new_num_cols;
        map_info_.height = new_num_rows;
        map_info_.origin.position.x -= pad_left * map_info_.resolution;
        map_info_.origin.position.y -= pad_top * map_info_.resolution;
    }

    nav_msgs::OccupancyGrid getMapAsRosMsg(const ros::Time& timestamp, const std::string& frame_id) {
        nav_msgs::OccupancyGrid msg;
        msg.header.seq = seq_++;
        msg.header.stamp = timestamp;
        msg.header.frame_id = frame_id;
        msg.info = map_info_;

        std::vector<int8_t> occupancy_belief(num_rows_ * num_cols_);

        for (int r = 0; r < num_rows_; ++r) {
            for (int c = 0; c < num_cols_; ++c) {
                double probability = logOddsRatioToProbability(log_odds_ratio_occupancy_grid_map_[r][c]);
                occupancy_belief[r * num_cols_ + c] = static_cast<int8_t>(100 * probability);
            }
        }

        msg.data = occupancy_belief;
        return msg;
    }

    std::vector<std::pair<int, int>> bresenham(int x0, int y0, int x1, int y1) {
        std::vector<std::pair<int, int>> points;
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;

        while (true) {
            points.emplace_back(x0, y0);
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx) {
                err += dx;
                y0 += sy;
            }
        }

        return points;
    }

    int getNumRows() const { return num_rows_; }
    int getNumCols() const { return num_cols_; }

private:
    double logOddsRatioToProbability(double log_odds) {
        return 1 - 1 / (1 + std::exp(log_odds));
    }

    int num_rows_;
    int num_cols_;
    double meters_per_cell_;
    double init_log_odds_;
    std::vector<std::vector<double>> log_odds_ratio_occupancy_grid_map_;
    nav_msgs::MapMetaData map_info_;
    int seq_;
};

class Communicator {
public:
    Communicator(const std::string& ns) : ns_(ns) {
        frame_sub_ = nh_.subscribe(ns_ + "/frame", 1, &Communicator::frameCallback, this);
    }

    bool hasFrameData() const {
        return !frame_buffer_.empty();
    }

    mm::Frame getNextFrame() {
        mm::Frame frame;
        if (hasFrameData()) {
            frame = frame_buffer_.front();
            frame_buffer_.erase(frame_buffer_.begin());
        }
        return frame;
    }

    size_t getFrameBufferSize() const {
        return frame_buffer_.size();
    }

private:
    void frameCallback(const mm::Frame::ConstPtr& msg) {
        frame_buffer_.push_back(*msg);
    }

    ros::NodeHandle nh_;
    std::string ns_;
    ros::Subscriber frame_sub_;
    std::vector<mm::Frame> frame_buffer_;
};

class GridMapper {
public:
    GridMapper(const std::string& robot1_namespace, const std::string& map_topic)
    : communicator_(robot1_namespace), map_(50, 50, 0.05, {0, 0, 0}) {
        grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(robot1_namespace + map_topic, 1);
    }

    void processFrame(const mm::Frame& frame_msg) {
        geometry_msgs::Point pose = frame_msg.pose;
        auto [pose_row, pose_col] = map_.cartesianToGridCoords(pose.x, pose.y);

        int min_row = std::numeric_limits<int>::max();
        int min_col = std::numeric_limits<int>::max();
        int max_row = std::numeric_limits<int>::min();
        int max_col = std::numeric_limits<int>::min();

        pcl::PointCloud<pcl::PointXYZ> cloud;
        pcl::fromROSMsg(frame_msg.cloud_2d, cloud);

        for (const auto& point : cloud.points) {
            auto [row, col] = map_.cartesianToGridCoords(point.x, point.y);
            min_row = std::min(min_row, row);
            min_col = std::min(min_col, col);
            max_row = std::max(max_row, row);
            max_col = std::max(max_col, col);
        }

        if (min_row < 0 || min_col < 0 || max_row >= map_.getNumRows() || max_col >= map_.getNumCols()) {
            map_.grow(min_row, min_col, max_row, max_col);
            std::tie(pose_row, pose_col) = map_.cartesianToGridCoords(pose.x, pose.y);
        }

        for (const auto& point : cloud.points) {
            auto [row, col] = map_.cartesianToGridCoords(point.x, point.y);
            auto cells = map_.bresenham(pose_row, pose_col, row, col);
            for (const auto& cell : cells) {
                map_.updateLogOddsRatio(cell.first, cell.second, -0.4);
            }
            map_.updateLogOddsRatio(row, col, 0.85);
        }
    }

    void publishGridMap() {
        ros::Time timestamp = ros::Time::now();
        std::string frame_id = "map";
        nav_msgs::OccupancyGrid grid_msg = map_.getMapAsRosMsg(timestamp, frame_id);
        grid_pub_.publish(grid_msg);
    }

    Communicator& getCommunicator() {
        return communicator_;
    }

private:
    ros::NodeHandle nh_;
    Communicator communicator_;
    OccupancyGridMap map_;
    ros::Publisher grid_pub_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gridmapper_node", ros::init_options::AnonymousName);


    ros::NodeHandle private_nh("~");
    std::string ns, map_topic;
    private_nh.getParam("robot_namespace", ns);
    private_nh.getParam("map_topic", map_topic);

    std::cout << ns + map_topic << std::endl;

    GridMapper grid_mapper(ns, map_topic);

    ros::Rate rate(100);
    ros::Duration(1.0).sleep();  // for runner init

    while (ros::ok()) {
        if (grid_mapper.getCommunicator().hasFrameData()) {
            mm::Frame frame_msg = grid_mapper.getCommunicator().getNextFrame();
            ROS_INFO("left frame element : %lu", grid_mapper.getCommunicator().getFrameBufferSize());
            if (!frame_msg.cloud_2d.data.empty()) {
                double st = ros::Time::now().toSec();
                grid_mapper.processFrame(frame_msg);
                grid_mapper.publishGridMap();
                double et = ros::Time::now().toSec();
                ROS_INFO("%f", et - st);
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
