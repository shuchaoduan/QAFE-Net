import cv2
import numpy as np


def gen_an_aug(results, sigma=1):
        all_kps = results['keypoint']
        kp_shape = all_kps.shape
        all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)
        img_h, img_w = results['img_shape']
        num_frame = kp_shape[0]

        imgs = []
        for i in range(num_frame):
            sigma = sigma
            kps = all_kps[i, :]
            kpscores = all_kpscores[i,:]
            max_values = np.ones(kpscores.shape, dtype=np.float32)
            hmap = generate_heatmap(img_h, img_w, kps, sigma, max_values)
            combined_map = np.sum(hmap, axis=0)
            # Apply Gaussian blur to smooth the combined heatmap
            smoothed_heatmap = cv2.GaussianBlur(combined_map, (21, 21), sigmaX=0)
            # Normalize the smoothed heatmap
            smoothed_heatmap /= np.max(smoothed_heatmap)
            # Convert the smoothed heatmap to an image
            smoothed_heatmap = cv2.applyColorMap(np.uint8(255 * smoothed_heatmap), cv2.COLORMAP_JET)
            # Display the heatmap
            # cv2.imshow('Heatmap', blurred_heatmap)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            imgs.append(smoothed_heatmap)
        return imgs

def generate_heatmap(img_h, img_w, kps, sigma, max_values, with_kp =True):
    #Generate pseudo heatmap for all keypoints in one frame
    heatmaps = []
    if with_kp:
        num_kp = kps.shape[0]

        for i in range(1, num_kp):
            heatmap = generate_a_heatmap(img_h, img_w, kps[i, :],
                                         sigma, max_values[i])
            heatmaps.append(heatmap)

    return np.stack(heatmaps, axis=0)

def generate_a_heatmap(img_h, img_w, centers,  sigma, max_values,):
    heatmap = np.zeros([img_h, img_w], dtype=np.float32)
    if len(centers) == 2:
        mu_x, mu_y = centers[0].astype(np.float32), centers[1].astype(np.float32)
        st_x = max(int(mu_x - 5), 0)
        ed_x = min(int(mu_x + 5) + 1, img_w)
        st_y = max(int(mu_y - 5), 0)
        ed_y = min(int(mu_y + 5) + 1, img_h)
        x = np.arange(st_x, ed_x, 1, np.float32)
        y = np.arange(st_y, ed_y, 1, np.float32)
        y = y[:, None]

        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
        patch = patch * max_values
        a = heatmap[st_y:ed_y, st_x:ed_x]
        b = np.maximum(a, patch)
        heatmap[st_y:ed_y, st_x:ed_x] = b

    return heatmap / np.max(heatmap)

if __name__=='__main__':
    item = {}
    item['keypoint'] = 'landmark coordinates'
    item['img_shape'] = (256, 256)
    heatmap_set = gen_an_aug(item, sigma=1)