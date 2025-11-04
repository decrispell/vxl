# brl/bseg/CMakeLists.txt
# main package for segmentation-related code and applications
# Mutual dependencies: brip --> sdet --> (vpro,segv)

# float image operations
add_subdirectory(brip)

# segmentation feature detection library
add_subdirectory(sdet)

# segmentation view library
add_subdirectory(segv)

# segmentation application executables
add_subdirectory(sbin)

# background modeling
add_subdirectory(bbgm)

# voxel world modeling
add_subdirectory(bvxm)

# Lidar modeling (segmentation and meshing)
add_subdirectory(bmdl)

# recognition library - uses background models of bbgm and/or bvxm
add_subdirectory(brec)

# octree
add_subdirectory(boct)

# updated, gpu compatible octree based modeling
add_subdirectory(boxm2)

# semi-global matching
add_subdirectory(bsgm)

# voxel processing library
add_subdirectory(bvpl)

# Brown affine patch library, SIFT keypoint & descriptor
add_subdirectory(bapl)

#multi gpu library
add_subdirectory(boxm2_multi)

#space-time modeling
add_subdirectory(bstm)
add_subdirectory(bstm_multi)

#the bridge lib to convert between bvxm and boxm2 scenes
add_subdirectory(bvxm_bridge)

#event trigger library
add_subdirectory(betr)

# VSI conflation tools
add_subdirectory(vcon)

# appearance modeling library
add_subdirectory(baml)
