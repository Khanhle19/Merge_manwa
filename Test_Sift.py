import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def find_template_position(large_image_path, small_image_path, output_dir=None):
    """
    Find position of small image in large image using SIFT
    
    Parameters:
    - large_image_path: Path to large image
    - small_image_path: Path to small image (template)
    - output_dir: Directory to save debug images (optional)
    
    Returns:
    - Position (x, y) of top-left corner of small image in large image
    - Confidence score
    """
    print("\n=== Finding Template Position with SIFT ===")
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load images
    start_time = time.time()
    print(f"Loading images...")
    
    large_image = cv2.imread(large_image_path)
    small_image = cv2.imread(small_image_path)
    
    if large_image is None or small_image is None:
        print(f"Error: Could not load one or both images.")
        print(f"Large image path: {large_image_path}")
        print(f"Small image path: {small_image_path}")
        return None, 0
    
    print(f"Large image dimensions: {large_image.shape[1]}x{large_image.shape[0]}")
    print(f"Small image dimensions: {small_image.shape[1]}x{small_image.shape[0]}")
    
    # Convert to grayscale
    large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    print("Finding keypoints and descriptors...")
    kp_small, des_small = sift.detectAndCompute(small_gray, None)
    kp_large, des_large = sift.detectAndCompute(large_gray, None)
    
    print(f"Found {len(kp_small)} keypoints in small image")
    print(f"Found {len(kp_large)} keypoints in large image")
    
    # Check if enough keypoints were found
    if len(kp_small) < 10 or len(kp_large) < 10:
        print("Not enough keypoints found in one or both images")
        return None, 0
    
    # Match descriptors using FLANN
    print("Matching descriptors...")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_small, des_large, k=2)
    
    # Apply Lowe's ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"Found {len(good_matches)} good matches out of {len(matches)} total matches")
    
    if len(good_matches) < 10:
        print("Not enough good matches found")
        return None, 0
    
    # Extract matched keypoints
    src_pts = np.float32([kp_small[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    print("Computing homography...")
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if homography is None:
        print("Could not find homography")
        return None, 0
    
    # Calculate confidence
    inliers = np.sum(mask)
    confidence = inliers / len(good_matches)
    print(f"Found {inliers} inliers out of {len(good_matches)} good matches (confidence: {confidence:.3f})")
    
    # Get corners of small image
    h, w = small_gray.shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    # Transform corners to find position in large image
    transformed_corners = cv2.perspectiveTransform(corners, homography)
    
    # Calculate top-left corner position
    top_left_x = int(transformed_corners[0][0][0])
    top_left_y = int(transformed_corners[0][0][1])
    
    # Calculate centroid as alternative position estimate
    centroid_x = int(np.mean([transformed_corners[i][0][0] for i in range(4)]))
    centroid_y = int(np.mean([transformed_corners[i][0][1] for i in range(4)]))
    
    print(f"Position found: top-left corner at ({top_left_x}, {top_left_y})")
    print(f"Centroid position: ({centroid_x}, {centroid_y})")
    
    if output_dir:
        # Create visualization images
        print("Creating visualizations...")
        
        # Draw keypoints
        small_keypoints = cv2.drawKeypoints(small_image, kp_small, None, color=(0, 255, 0))
        large_keypoints = cv2.drawKeypoints(large_image, kp_large, None, color=(0, 255, 0))
        
        # Draw matches
        matches_img = cv2.drawMatches(small_image, kp_small, large_image, kp_large, 
                                     good_matches[:50], None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Draw result
        result_img = large_image.copy()
        
        # Draw polygon around found region
        pts = transformed_corners.reshape(-1, 2).astype(int)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 3)
        
        # Draw corner points
        for i in range(4):
            cv2.circle(result_img, (int(transformed_corners[i][0][0]), int(transformed_corners[i][0][1])), 
                      10, (0, 0, 255), -1)
        
        # Add text labels for corners
        corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i in range(4):
            x = int(transformed_corners[i][0][0])
            y = int(transformed_corners[i][0][1])
            cv2.putText(result_img, corner_labels[i], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Draw centroid
        cv2.circle(result_img, (centroid_x, centroid_y), 15, (255, 0, 0), -1)
        cv2.putText(result_img, "Centroid", (centroid_x, centroid_y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Add rectangle representing exact placement
        cv2.rectangle(result_img, (top_left_x, top_left_y), 
                     (top_left_x + small_image.shape[1], top_left_y + small_image.shape[0]), 
                     (255, 255, 0), 3)
        
        # Save images
        cv2.imwrite(os.path.join(output_dir, "small_keypoints.jpg"), small_keypoints)
        cv2.imwrite(os.path.join(output_dir, "large_keypoints.jpg"), large_keypoints)
        cv2.imwrite(os.path.join(output_dir, "matches.jpg"), matches_img)
        cv2.imwrite(os.path.join(output_dir, "result.jpg"), result_img)
        
        # Save composite image showing placement
        composite = large_image.copy()
        
        # Check if the template position is valid
        valid_region = (top_left_x >= 0 and top_left_y >= 0 and 
                       top_left_x + small_image.shape[1] <= large_image.shape[1] and 
                       top_left_y + small_image.shape[0] <= large_image.shape[0])
        
        if valid_region:
            # Create semi-transparent overlay of small image at detected position
            overlay = composite[top_left_y:top_left_y+small_image.shape[0], 
                               top_left_x:top_left_x+small_image.shape[1]].copy()
            cv2.addWeighted(small_image, 0.5, overlay, 0.5, 0, overlay)
            composite[top_left_y:top_left_y+small_image.shape[0], 
                     top_left_x:top_left_x+small_image.shape[1]] = overlay
        else:
            print("Warning: Template position is outside image bounds, skipping overlay")
            
            # Calculate valid region for partial overlay if possible
            start_y = max(0, top_left_y)
            start_x = max(0, top_left_x)
            end_y = min(large_image.shape[0], top_left_y + small_image.shape[0])
            end_x = min(large_image.shape[1], top_left_x + small_image.shape[1])
            
            if start_y < end_y and start_x < end_x:  # There is some overlap
                # Calculate corresponding region in small image
                small_start_y = start_y - top_left_y if top_left_y < 0 else 0
                small_start_x = start_x - top_left_x if top_left_x < 0 else 0
                small_end_y = small_start_y + (end_y - start_y)
                small_end_x = small_start_x + (end_x - start_x)
                
                try:
                    # Get the overlapping regions
                    small_region = small_image[small_start_y:small_end_y, small_start_x:small_end_x]
                    large_region = composite[start_y:end_y, start_x:end_x]
                    
                    # Check sizes match
                    if small_region.shape == large_region.shape:
                        # Create blended overlay
                        cv2.addWeighted(small_region, 0.5, large_region, 0.5, 0, large_region)
                        composite[start_y:end_y, start_x:end_x] = large_region
                        print("Created partial overlay for template")
                except Exception as e:
                    print(f"Error creating partial overlay: {e}")
        
        x1 = max(0, top_left_x)
        y1 = max(0, top_left_y)
        x2 = min(large_image.shape[1], top_left_x + small_image.shape[1])
        y2 = min(large_image.shape[0], top_left_y + small_image.shape[0])
        cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        cv2.imwrite(os.path.join(output_dir, "composite.jpg"), composite)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return (top_left_x, top_left_y), confidence

def test_with_visualization():
    large_image_path = r"E:\Manwa\dungeon-reset\raw2\inpainted\c9-01.png"
    small_image_path = r"E:\Manwa\dungeon-reset\vn1\c9\014.jpg"
    
    output_dir = "template_matching_results"
    
    position, confidence = find_template_position(large_image_path, small_image_path, output_dir)
    
    if position:
        print(f"\nResults:")
        print(f"Template position: ({position[0]}, {position[1]})")
        print(f"Confidence: {confidence:.3f}")
        print(f"Visualization images saved in {output_dir}/")
    else:
        print("Could not find template in image")

if __name__ == "__main__":
    test_with_visualization()