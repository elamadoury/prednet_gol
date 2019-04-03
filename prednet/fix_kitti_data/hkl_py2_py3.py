import hickle as hkl

data_path = "../prednet/kitti_data/"
# original_hkl = hkl.load(data_path + "sources_test_27.hkl")
# hkl.dump(original_hkl, data_path + "sources_test_36.hkl", protocol=0)
# original_hkl = hkl.load(data_path + "sources_train_27.hkl")
# hkl.dump(original_hkl, data_path + "sources_train_36.hkl", protocol=0)
# original_hkl = hkl.load(data_path + "sources_val_27.hkl")
# hkl.dump(original_hkl, data_path + "sources_val_36.hkl", protocol=0)

original_hkl = hkl.load(data_path + "X_test_27.hkl")
hkl.dump(original_hkl, data_path + "X_test_36.hkl")
original_hkl = hkl.load(data_path+ "X_train_27.hkl")
hkl.dump(original_hkl, data_path + "X_train_36.hkl")
original_hkl = hkl.load(data_path + "X_val_27.hkl")
hkl.dump(original_hkl, data_path + "X_val_36.hkl")