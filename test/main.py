import cv2
import numpy as np
import random
import math

from sticker import Sticker

def heuristic(stickers, sheet_width, sheet_height):
    overlap_penalty = 0
    spacing_penalty = 0
    edge_penalty = 0

    # Check for overlap between stickers
    for i in range(len(stickers)):
        for j in range(i + 1, len(stickers)):
            if contours_overlap(stickers[i].contour, stickers[j].contour):
                overlap_penalty += 100  # Large penalty for overlap

        # Calculate distance to other stickers (minimize close proximity)
        for j in range(i + 1, len(stickers)):
            spacing_penalty += 1 / (min_distance_between_contours(stickers[i].contour, stickers[j].contour) + 1)

        # Calculate distance to the edges of the sheet
        edge_penalty += distance_to_edge(stickers[i], sheet_width, sheet_height)

    return overlap_penalty + spacing_penalty + edge_penalty

def distance_to_edge(sticker, sheet_width, sheet_height):
    contours = sticker.contour

    # Get at most points
    leftmost = contours[contours[:, 0].argmin()]
    rightmost = contours[contours[:, 0].argmax()]
    topmost = contours[contours[:, 1].argmin()]
    bottommost = contours[contours[:, 1].argmax()]

    left_dist = leftmost
    right_dist = sheet_width - rightmost
    top_dist = topmost
    bottom_dist = sheet_height - bottommost

    # Normalize distances
    return (1 / left_dist) + (1 / right_dist) + (1 / top_dist) + (1 / bottom_dist)

def min_distance_between_contours(contour1, contour2):
    min_distance = float('inf')

    for point1 in contour1:
        for point2 in contour2:
            distance = np.linalg.norm(point1 - point2)
            if distance < min_distance: min_distance = distance

    return min_distance

def contours_overlap(contour1, contour2):
    for point in contour1:
        if cv2.pointPolygonTest(contour2, tuple(point), False) >= 0:
            return True
        
    for point in contour2:
        if cv2.pointPolygonTest(contour1, tuple(point), False) >= 0:
            return True
        
    return False

# Generate a neighbor arrangement by moving or rotating a sticker
def generate_neighbor(stickers):
    neighbor = [Sticker(sticker.contour.copy(), sticker.x, sticker.y) for sticker in stickers]
    random_sticker = random.choice(neighbor)

    if random.random() < 0.5:
        random_sticker.move(random.randint(-10, 10), random.randint(-10, 10))
    else:
        angle = random.choice([0, 90, 180, 270])
        random_sticker.rotate(angle)

    return neighbor

def simulated_annealing(stickers, sheet_width, sheet_height, initial_temperature=1000, cooling_rate=0.95):
    current_stickers = stickers
    current_energy = heuristic(current_stickers, sheet_width, sheet_height)
    best_stickers = stickers
    best_energy = current_energy

    temperature = initial_temperature

    while temperature > 1:
        neighbor_stickers = generate_neighbor(current_stickers)
        neighbor_energy = heuristic(neighbor_stickers, sheet_width, sheet_height)

        # Accept neighbor if it's better, or with some probability if worse
        if neighbor_energy < current_energy or random.uniform(0, 1) < math.exp((current_energy - neighbor_energy) / temperature):
            current_stickers = neighbor_stickers
            current_energy = neighbor_energy

            if current_energy < best_energy:
                best_stickers = current_stickers
                best_energy = current_energy

        temperature *= cooling_rate

    return best_stickers

def visualize_result(stickers, sheet_width, sheet_height, output_image_path):
    output_image = np.ones((sheet_height, sheet_width, 3), dtype=np.uint8) * 255

    light_green = (144, 238, 144)

    for sticker in stickers:
        contour = sticker.contour + np.array([sticker.x, sticker.y])
        cv2.drawContours(output_image, [contour], -1, light_green, 2)

    cv2.imwrite(output_image_path, output_image)


if __name__ == '__main__':
    stickers = ["stickers/mario.png", "stickers/link.png", "stickers/pikachu.png", "stickers/kirby.png"]
    output_image_path = 'output.png'

    sheet_width = 800
    sheet_height = 600
    stickers = []

    for sticker_file in stickers:
        image = cv2.imread(sticker_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image from path: {sticker_file}")

        sticker = Sticker(image, random.randint(0, sheet_width), random.randint(0, sheet_height))

        # Check if sticker fits within the sheet
        if sticker.width > sheet_width or sticker.height > sheet_height:
            raise ValueError(f"Could not fit image in sheet")

        stickers.append(sticker)

    optimized_stickers = simulated_annealing(stickers, sheet_width, sheet_height)
    # visualize_result(optimized_stickers, sheet_width, sheet_height, output_image_path)
