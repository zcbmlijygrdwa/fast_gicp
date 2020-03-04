#include <chrono>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_gicp_st.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <guik/viewer/light_viewer.hpp>

/**
 * @brief main
 */
int main(int argc, char** argv) {
  if(argc < 3) {
    std::cout << "usage: gicp_align target_pcd source_pcd" << std::endl;
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if(pcl::io::loadPCDFile(argv[1], *target_cloud)) {
    std::cerr << "failed to open " << argv[1] << std::endl;
    return 1;
  }
  if(pcl::io::loadPCDFile(argv[2], *source_cloud)) {
    std::cerr << "failed to open " << argv[2] << std::endl;
    return 1;
  }

  // downsampling
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.01f, 0.01f, 0.01f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*filtered);
  target_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*filtered);
  source_cloud = filtered;

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << target_cloud->size() << "[pts]" << std::endl;

  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputSource(source_cloud);
  gicp.setInputTarget(target_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
  gicp.align(*aligned);

  std::cout << gicp.getFinalTransformation() << std::endl;

  auto viewer = guik::LightViewer::instance();
  viewer->update_drawable("source", std::make_shared<glk::PointCloudBuffer>(source_cloud), guik::FlatColor(Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f), gicp.getFinalTransformation()));
  viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(target_cloud), guik::FlatColor(Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f)));
  viewer->spin();

  return 0;
}