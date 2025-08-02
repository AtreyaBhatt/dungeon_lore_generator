import numpy as np
from PIL import Image, ImageDraw, ImageFont
import noise
import random

class FantasyMapGenerator:
    def __init__(self, width=800, height=600, seed=None):
        self.width = width
        self.height = height
        self.seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(self.seed)
        
        # Biome parameters
        self.biomes = {
            'deep_ocean': (50, 70, 150),
            'ocean': (70, 100, 200),
            'shore': (220, 220, 150),
            'sand': (240, 240, 180),
            'grass': (100, 180, 80),
            'forest': (40, 120, 60),
            'jungle': (30, 100, 50),
            'mountain': (120, 120, 120),
            'snow': (240, 240, 240),
            'swamp': (80, 120, 80),
            'desert': (240, 220, 120),
            'tundra': (200, 220, 220)
        }
        
        # Settlement types
        self.settlements = {
            'city': {'color': (255, 0, 0), 'size': 8, 'symbol': '★'},
            'town': {'color': (200, 100, 0), 'size': 6, 'symbol': '■'},
            'village': {'color': (150, 150, 0), 'size': 4, 'symbol': '●'},
            'landmark': {'color': (0, 150, 150), 'size': 5, 'symbol': '▲'}
        }
        
        # Generate the map
        self.generate_map()
    
    def generate_map(self):
        # Create elevation map using Perlin noise
        self.elevation = np.zeros((self.height, self.width))
        self.moisture = np.zeros((self.height, self.width))
        
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        for y in range(self.height):
            for x in range(self.width):
                nx = x/self.width - 0.5
                ny = y/self.height - 0.5
                
                # Elevation
                e = 1.0 * noise.pnoise2(1 * nx, 1 * ny, octaves=octaves, persistence=persistence, 
                                      lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=self.seed)
                e += 0.5 * noise.pnoise2(2 * nx, 2 * ny, octaves=octaves, persistence=persistence, 
                                       lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=self.seed+1)
                e += 0.25 * noise.pnoise2(4 * nx, 4 * ny, octaves=octaves, persistence=persistence, 
                                        lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=self.seed+2)
                
                e = (e + 1) / 2
                self.elevation[y][x] = e
                
                # Moisture
                m = noise.pnoise2(nx, ny, octaves=4, persistence=0.5, 
                                lacunarity=2.0, repeatx=1024, repeaty=1024, base=self.seed+3)
                m = (m + 1) / 2
                self.moisture[y][x] = m
        
        # Normalize
        self.elevation = (self.elevation - self.elevation.min()) / (self.elevation.max() - self.elevation.min())
        self.moisture = (self.moisture - self.moisture.min()) / (self.moisture.max() - self.moisture.min())
        
        # Create biome map
        self.biome_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for y in range(self.height):
            for x in range(self.width):
                e = self.elevation[y][x]
                m = self.moisture[y][x]
                
                if e < 0.3:
                    if e < 0.1:
                        biome = 'deep_ocean'
                    else:
                        biome = 'ocean'
                elif e < 0.35:
                    biome = 'shore'
                else:
                    if e > 0.8:
                        if e > 0.9:
                            biome = 'snow'
                        else:
                            biome = 'mountain'
                    else:
                        if m < 0.3:
                            if e < 0.5:
                                biome = 'sand'
                            else:
                                biome = 'desert'
                        elif m < 0.6:
                            if e < 0.6:
                                biome = 'grass'
                            else:
                                biome = 'forest'
                        else:
                            if e < 0.5:
                                biome = 'swamp'
                            else:
                                biome = 'jungle'
                
                self.biome_map[y][x] = self.biomes[biome]
        
        # Add rivers
        self.add_rivers()
        
        # Add settlements
        self.add_settlements()
    
    def add_rivers(self):
        num_rivers = random.randint(5, 12)
        
        for _ in range(num_rivers):
            while True:
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)
                if self.elevation[y][x] > 0.7:
                    break
            
            river_length = 0
            max_length = random.randint(50, 300)
            
            while river_length < max_length:
                min_elevation = self.elevation[y][x]
                next_x, next_y = x, y
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx = x + dx
                        ny = y + dy
                        
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.elevation[ny][nx] < min_elevation:
                                min_elevation = self.elevation[ny][nx]
                                next_x, next_y = nx, ny
                
                if next_x == x and next_y == y:
                    break
                
                width = max(1, int((max_length - river_length) / 50))
                for wy in range(-width, width+1):
                    for wx in range(-width, width+1):
                        if 0 <= y+wy < self.height and 0 <= x+wx < self.width:
                            if self.elevation[y+wy][x+wx] > 0.3:
                                dist = (wx**2 + wy**2)**0.5
                                if dist <= width:
                                    blend = 0.7 - 0.3 * (dist / width)
                                    water_color = self.biomes['ocean']
                                    current_color = self.biome_map[y+wy][x+wx]
                                    r = int(water_color[0] * blend + current_color[0] * (1 - blend))
                                    g = int(water_color[1] * blend + current_color[1] * (1 - blend))
                                    b = int(water_color[2] * blend + current_color[2] * (1 - blend))
                                    self.biome_map[y+wy][x+wx] = (r, g, b)
                
                x, y = next_x, next_y
                river_length += 1
    
    def add_settlements(self):
        self.settlement_data = []
        
        # Generate random names
        prefixes = ['North', 'South', 'East', 'West', 'New', 'Old', 'Port', 'Lake', 'River']
        suffixes = ['burg', 'ton', 'ville', 'ford', 'shire', 'haven', 'field', 'wood', 'rock']
        roots = ['Elm', 'Oak', 'Stone', 'Bright', 'Green', 'Fair', 'Wind', 'Gold', 'Silver']
        
        def generate_name():
            if random.random() < 0.3:
                return random.choice(prefixes) + ' ' + random.choice(roots) + random.choice(suffixes)
            else:
                return random.choice(roots) + random.choice(suffixes)
        
        possible_locations = []
        for y in range(self.height):
            for x in range(self.width):
                e = self.elevation[y][x]
                if 0.35 < e < 0.7:
                    possible_locations.append((x, y))
        
        if not possible_locations:
            return
        
        # Add cities
        num_capitals = random.randint(1, 3)
        for _ in range(num_capitals):
            x, y = random.choice(possible_locations)
            self.settlement_data.append({
                'type': 'city',
                'x': x,
                'y': y,
                'name': generate_name()
            })
            possible_locations = [(px, py) for (px, py) in possible_locations 
                                if (px-x)**2 + (py-y)**2 > 400]
        
        # Add towns
        num_towns = random.randint(5, 15)
        for _ in range(num_towns):
            if not possible_locations:
                break
            x, y = random.choice(possible_locations)
            self.settlement_data.append({
                'type': 'town',
                'x': x,
                'y': y,
                'name': generate_name()
            })
            possible_locations = [(px, py) for (px, py) in possible_locations 
                                if (px-x)**2 + (py-y)**2 > 100]
        
        # Add villages
        num_villages = random.randint(10, 25)
        for _ in range(num_villages):
            if not possible_locations:
                break
            x, y = random.choice(possible_locations)
            self.settlement_data.append({
                'type': 'village',
                'x': x,
                'y': y,
                'name': generate_name()
            })
            possible_locations = [(px, py) for (px, py) in possible_locations 
                                if (px-x)**2 + (py-y)**2 > 25]
        
        # Add landmarks
        num_landmarks = random.randint(3, 8)
        for _ in range(num_landmarks):
            if not possible_locations:
                break
            x, y = random.choice(possible_locations)
            landmarks = ['Dragon\'s Peak', 'Wizard\'s Tower', 'Ancient Ruins', 
                        'Forgotten Temple', 'Cursed Keep', 'Giant\'s Bones']
            self.settlement_data.append({
                'type': 'landmark',
                'x': x,
                'y': y,
                'name': random.choice(landmarks)
            })
            possible_locations = [(px, py) for (px, py) in possible_locations 
                                if (px-x)**2 + (py-y)**2 > 100]
    
    def draw_settlements(self, img):
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            symbol_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            symbol_font = ImageFont.load_default()
        
        for settlement in self.settlement_data:
            props = self.settlements[settlement['type']]
            x = settlement['x']
            y = settlement['y']
            
            # Draw symbol
            draw.text((x - props['size']//2, y - props['size']//2 - 5), 
                     props['symbol'], fill=props['color'], font=symbol_font)
            
            # Draw name
            bbox = draw.textbbox((0, 0), settlement['name'], font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle([(x - text_width//2 - 2, y + props['size'] + 5),
                          (x + text_width//2 + 2, y + props['size'] + text_height + 7)], 
                         fill=(255, 255, 255, 200))
            draw.text((x - text_width//2, y + props['size'] + 5), 
                     settlement['name'], fill=(0, 0, 0), font=font)
    
    def get_map_image(self):
        img = Image.fromarray(self.biome_map, 'RGB')
        self.draw_settlements(img)
        return img
    
    def save_map(self, filename='fantasy_map.png'):
        img = self.get_map_image()
        img.save(filename)
        print(f"Map saved to {filename}")

if __name__ == "__main__":
    map_gen = FantasyMapGenerator(width=800, height=600, seed=42)
    map_gen.save_map()
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 9))
        plt.imshow(map_gen.get_map_image())
        plt.axis('off')
        plt.title('Fantasy World Map')
        plt.show()
    except ImportError:
        print("Map saved as fantasy_map.png - open this file to view your map")