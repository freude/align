CFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g -O3
CXXFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g -O3
LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_nonfree -lopencv_features2d -lopencv_video -L$(HDF5_DIR)/lib -lhdf5_cpp -lhdf5

all: demo compute_mapping warp_image refine_mapping show_mapping chain_warps keypoint_matching_stats

demo: demo.o
	g++ -o $@ $^ $(LDFLAGS)

compute_mapping: compute_mapping.o opencv_hdf5.o
	g++ -o $@ $^ $(LDFLAGS)

refine_mapping: refine_mapping.o opencv_hdf5.o
	g++ -o $@ $^ $(LDFLAGS)

chain_warps: chain_warps.o opencv_hdf5.o
	g++ -o $@ $^ $(LDFLAGS)

show_mapping: show_mapping.o opencv_hdf5.o
	g++ -o $@ $^ $(LDFLAGS)

warp_image: warp_image.o opencv_hdf5.o
	g++ -o $@ $^ $(LDFLAGS)

keypoint_matching_stats: keypoint_matching_stats.o
	g++ -o $@ $^ $(LDFLAGS)

