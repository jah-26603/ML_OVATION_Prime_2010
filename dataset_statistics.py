import numpy as np
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(f):
    """Process a single file and return channel sums and squared sums"""
    img = np.load(f)
    return img.sum(axis=(1,2)), (img**2).sum(axis=(1,2))

if __name__ == '__main__':
    fp = r'D:\ml_aurora\paired_data\images'
    files = glob.glob(os.path.join(fp, '*.npy'))
    C, H, W = np.load(files[0]).shape
    
    # Use multiprocessing
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    
    # Aggregate results
    sum_ = np.zeros(C, dtype=np.float64)
    sum_sq = np.zeros(C, dtype=np.float64)
    
    for s, sq in results:
        sum_ += s
        sum_sq += sq
    
    pixel_count = H * W * len(files)
    mean = sum_ / pixel_count
    std = np.sqrt(sum_sq / pixel_count - mean**2)
    
    np.save('mean_stats.npy', mean)
    np.save('std_stats.npy', std)
    print(f"Mean: {mean}")
    print(f"Std: {std}")