import pygame
import random
import numpy as np
import string

# Set screen size and other constants
WIDTH, HEIGHT = 1000, 800
bullets = []
punch_effects = []
HUNGER_RATE = 0.1
MAX_HUNGER = 100
DAMAGE_BULLET = 30
DAMAGE_PUNCH = 20
REPRODUCE_HEALTH = 65
REPRODUCE_HUNGER = 65
PUNCH_RANGE = 30
PUNCH_COOLDOWN = 20
KNOCKBACK_FORCE = 20
SHOOTING_COOLDOWN = 20
DASH_DISTANCE = 100
DASH_COOLDOWN = 60
VOWELS = "AEIOU"
CONSONANTS = "".join([c for c in string.ascii_uppercase if c not in VOWELS])
epoch = 0

# Entity brain class

class EntityBrain():
    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # number of weight layers

        # Store weights and biases as lists
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            self.weights.append(np.random.rand(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def forward(self, x):
        a = x
        for i in range(self.num_layers - 1):  # all hidden layers
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.relu(z)
        # final layer → softmax
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        return self.softmax(z)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# Bullet class to represent bullets

class Bullet:
    def __init__(self, bx, by, bdir, speed=20, bounce=1):
        self.x = bx
        self.y = by
        self.dir = (bdir*(np.pi/180))
        self.speed = speed
        self.bounce = bounce

    def update(self):

        # Move the bullet
        self.x += np.cos(self.dir) * self.speed
        self.y += np.sin(self.dir) * self.speed

        # Bounce off walls
        if self.x <= 0 or self.x >= WIDTH:
            self.dir = np.pi - self.dir  # reverse X direction
            self.x = max(0, min(WIDTH, self.x))  # clamp inside screen
            self.bounce -= 1

        if self.y <= 0 or self.y >= HEIGHT:
            self.dir = -self.dir  # reverse Y direction
            self.y = max(0, min(HEIGHT, self.y))  # clamp inside screen
            self.bounce -= 1

# Punch effect class

class PunchEffect:
    def __init__(self, x, y, color, duration=10, size=20):
        self.x = x
        self.y = y
        self.color = color
        self.duration = duration  # frames the effect lasts
        self.size = size

    def update(self):
        self.duration -= 1
        # Optional: grow/shrink or fade
        self.size += 1
        self.color = (
            max(0, self.color[0]-20),
            max(0, self.color[1]-20),
            self.color[2]
        )

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x + 10), int(self.y + 10)), self.size)  
        # +10 centers it on the entity

# Name functions

def generate_id(length=5):
        id = ""
        start_with_consonant = random.choice([True, False])
    
        for i in range(length):
            if (i % 2 == 0 and start_with_consonant) or (i % 2 == 1 and not start_with_consonant):
                id += random.choice(CONSONANTS)
            else:
                id += random.choice(VOWELS)
        return id

def mutate_id(parent_id):
    """Create a modified version of the parent's name ID."""
    id = list(parent_id)
    
    # Randomly tweak letters
    for i in range(len(id)):
        if random.random() < 0.4:  # 40% chance to change a letter
            if id[i] in VOWELS:
                id[i] = random.choice(VOWELS)
            if id[i] in CONSONANTS:
                id[i] = random.choice(CONSONANTS)
    
    # Randomly remove a letter
    if random.random() < 0.1 and len(id) > 3:
        idx = random.randint(0, len(id)-1)
        id.pop(idx)
    
    # Randomly add a letter
    if random.random() < 0.1 and len(id) < 7:
        idx = random.randint(0, len(id))
        if idx % 2 == 0:
            id.insert(idx, random.choice(CONSONANTS))
        else:
            id.insert(idx, random.choice(VOWELS))
    
    return "".join(id)

epoch_name = generate_id(random.randint(4,6))
epoch_name += 'CENE'

# Main entity class

class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.l1_size = 12
        self.l2_size = 12
        self.brain = EntityBrain([11, self.l1_size, self.l2_size, 11]) # 11 inputs, 12 hidden neurons in layer 1 and 2 respectively, and 11 outputs
        self.speed = random.uniform(0,10)
        self.agility = random.randint(0,1)
        self.speedmax = random.randint(12,18)
        self.health = random.randint(75,125)
        self.hunger = 0
        self.age = 0
        self.reproduce_cooldown = 300 # frames until it has a child
        self.alive = True
        self.punch_cooldown = random.randint(55,65)
        self.shoot_cooldown = random.randint(55,80)
        self.dash_cooldown = random.randint(55,65)
        self.resistance = random.randint(0,1)
        self.punch_damage = 0
        self.aggro_bias = random.uniform(-0.5,0.5)
        self.family_id = random.randint(0,255)
        self.bullet_quality = random.randint(0,1)
        self.gestation = 0
        self.regen_factor = 0.5
        self.id = generate_id(5)
        self.vision = random.randint(150,300)
        self.vx = 0  # velocity in x
        self.vy = 0  # velocity in y
        self.knockback_decay = 0.8
        self.color = (255, 255, 255)
        self.is_survivor = False
        self.generation = 1
        self.reward = 0
        self.is_descendant = False

    def check_collision(self, other):
        size = 20
        return (
            abs(self.x - other.x) < size and
            abs(self.y - other.y) < size
        )

    def get_inputs(self, entities):
        nearest_enemy = None
        nearest_ally = None

        min_enemy_dist = float("inf")
        min_ally_dist = float("inf")

        enemy_dx = enemy_dy = 0
        ally_dx = ally_dy = 0

        nearby_count = 0

        for other in entities:
            if other is self or not other.alive:
                continue

            dx = other.x - self.x
            dy = other.y - self.y
            dist = np.hypot(dx, dy)

            if dist < self.vision:
                nearby_count += 1

                relatedness = abs(other.family_id - self.family_id) # 0 = same family, 1 = somewhat related, 10+ = probably unrelated
                relatedness = min(relatedness, 256 - relatedness) # wrap around high values
                tolerance = int((1 - (self.aggro_bias + 0.5)) * 100) # high aggro = less tolerant of differences, low aggro = more tolerant

                # Separate allies and enemies (race detection)
                if relatedness <= tolerance:
                    if dist < min_ally_dist:
                        min_ally_dist = dist
                        nearest_ally = other
                        ally_dx, ally_dy = dx, dy
                else:
                    if dist < min_enemy_dist:
                        min_enemy_dist = dist
                        nearest_enemy = other
                        enemy_dx, enemy_dy = dx, dy

        # Normalize
        def safe_norm(value, max_val):
            return value / max_val if max_val != 0 else 0

        # Enemy info
        enemy_dir_x = safe_norm(enemy_dx, WIDTH)
        enemy_dir_y = safe_norm(enemy_dy, HEIGHT)
        enemy_dist = safe_norm(min_enemy_dist if nearest_enemy else 0, WIDTH)

        # Ally info
        ally_dir_x = safe_norm(ally_dx, WIDTH)
        ally_dir_y = safe_norm(ally_dy, HEIGHT)
        ally_dist = safe_norm(min_ally_dist if nearest_ally else 0, WIDTH)

        # Density (how crowded area is)
        density = min(1, nearby_count / 10)

        inputs = np.array([
            enemy_dir_x,    # 0
            enemy_dir_y,    # 1
            enemy_dist,     # 2
            ally_dist,      # 3
            self.hunger,    # 4
            self.health,    # 5
            density,        # 6
            self.speed,     # 7
            self.reward,    # 8
            ally_dir_x,     # 9
            ally_dir_y      # 10
        ])

        # Direction for shooting (enemy priority)
        nearest_dir = None
        if nearest_enemy:
            nearest_dir = np.arctan2(enemy_dy, enemy_dx)

        return inputs, nearest_dir

    def update(self, entities):
        if self.alive == False:
            return
        inputs, nearest_dir = self.get_inputs(entities)
        output = self.brain.forward(inputs)

        self.x += self.vx
        self.y += self.vy

        # Movement based on outputs
        self.x += (self.speed + self.agility) * (output[3] - output[2])
        self.y += (self.speed + self.agility) * (output[1] - output[0])

        # Dash action
        dash_trigger = output[9]
        dash_radians = output[10] * 2 * np.pi
        self.dash = False

        if self.dash_cooldown <= 0 and dash_trigger > 0.5:
            self.dash = True
            self.reward += 0.1 # small dopamine dose
            dvx = np.cos(dash_radians) * (DASH_DISTANCE + self.agility)
            dvy = np.sin(dash_radians) * (DASH_DISTANCE + self.agility)

            self.vx += dvx
            self.vy += dvy

            # Knockback nearby entities
            for other in entities:
                if other is self or not other.alive:
                    continue
                dist = np.hypot(other.x - self.x, other.y - self.y)
                if dist < 25:  # collision radius
                    self.reward += 0.25 # another treat
                    kx = (other.x - self.x) / (dist + 1e-5)
                    ky = (other.y - self.y) / (dist + 1e-5)
                    other.vx += kx * KNOCKBACK_FORCE
                    other.vy += ky * KNOCKBACK_FORCE

            self.dash_cooldown = DASH_COOLDOWN

        self.dash_cooldown -= 1
        self.dash_cooldown = max(0, self.dash_cooldown)
        
        # Decay knockback
        self.vx *= self.knockback_decay
        self.vy *= self.knockback_decay

        if output[4] >= 0.2 and self.speed <= self.speedmax + self.agility:
            self.speed += self.agility + 1

        # Shoot action

        shoot = False
        if output[5] * self.aggro_bias >= ( 0.5 * output[7]) and self.shoot_cooldown == 0 and nearest_dir is not None:
            new_bullet = Bullet(self.x, self.y, np.degrees(nearest_dir))
            new_bullet.bounce = 3 + self.bullet_quality
            new_bullet.shooter = self
            bullets.append(new_bullet)
            self.shoot_cooldown = SHOOTING_COOLDOWN
            shoot = True

        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1 + (output[5])

        # Punch action

        punch = False
        if self.punch_cooldown <= 0 and output[6] * self.aggro_bias > output[7]:
            for other in entities:
                if other is self or not other.alive:
                    continue
                dist = np.hypot(other.x - self.x, other.y - self.y)
                if dist < PUNCH_RANGE:
                    other.health -= (DAMAGE_PUNCH + self.punch_damage) * (1 + self.speed/50)
                    dx = other.x - self.x
                    dy = other.y - self.y
                    dist = np.hypot(dx, dy) + 1e-5

                    other.vx += (dx / dist) * KNOCKBACK_FORCE
                    other.vy += (dy / dist) * KNOCKBACK_FORCE

                    punch_effects.append(PunchEffect(other.x, other.y, (255, 255, 0)))

                    punch = True

                    # Reward attacker
                    self.hunger = max(0, self.hunger - DAMAGE_PUNCH * 0.5)
                    self.health = min(100, self.health + DAMAGE_PUNCH * 0.3)
                    self.reward += 1

        if punch == True:
            self.punch_cooldown = PUNCH_COOLDOWN
        elif  self.punch_cooldown > 0:
            self.punch_cooldown -= 1 + (output[6])

        self.age += 1
        if self.reproduce_cooldown > 0:
            self.reproduce_cooldown -= 1
        if self.age % 60 == 0:
            self.reward += 0.5

        # Reproduction

        if (self.health > REPRODUCE_HEALTH and self.hunger < REPRODUCE_HUNGER
            and self.reproduce_cooldown <= self.gestation):
            self.reward += 7.5
            # have children
            for i in range(random.randint(1, 2 + int(output[7] * 2))):
                child_x = self.x + random.randint(20, 56) * random.choice([-1, 1])
                child_y = self.y + random.randint(20, 56) * random.choice ([-1, 1])
                child = Entity(child_x, child_y)
        
                # Inherit characteristics w/ small mutations

                mutf1, mutf2, mutf3, mutf4, mutf5, mutf6, mutf7 = [random.uniform(-0.5, 0.5) for i in range(7)] # set mutation factors

                # Weights and biases
                for i in range(len(self.brain.weights)):
                    child.brain.weights[i] = self.brain.weights[i] + np.random.normal(0, 0.05, self.brain.weights[i].shape)
                    child.brain.biases[i] = self.brain.biases[i] + np.random.normal(0, 0.05, self.brain.biases[i].shape)

                # Brain topology

                add_chance = 0.05 # 5% chance to add new neuron
                remove_chance = 0.02 # 2% chance to remove existing neuron

                if random.uniform(0, 1) <= add_chance:
                    child.l1_size += 1
                if random.uniform(0, 1) <= remove_chance and child.l2_size > 0:
                    child.l1_size -= 1

                if random.uniform(0, 1) <= add_chance:
                    child.l2_size += 1
                if random.uniform(0, 1) <= remove_chance and child.l2_size > 0:
                    child.l2_size -= 1

                # Other stuff
                child.resistance = self.resistance + mutf1
                child.health = self.health + mutf2
                child.speedmax = self.speedmax - (mutf2 + mutf1)
                child.agility = self.agility - (mutf1 * 2)
                child.punch_damage = self.punch_damage + (mutf3 * 2)
                child.punch_cooldown = self.punch_cooldown - (mutf3 * 2)
                child.shoot_cooldown = self.shoot_cooldown - (mutf3 * 2)
                child.aggro_bias = self.aggro_bias + mutf4
                child.bullet_quality = self.bullet_quality - int(mutf4)
                child.regen_factor += mutf7
                child.gestation = self.gestation + int(mutf5 * 10)
                child.vision = self.vision + (mutf6 * 25)
                child.id = mutate_id(self.id)
                if np.random.randint(1,10) == 1:
                    child.family_id = self.family_id + (np.random.randint(-1,1) * random.randint(0,1))
                else:
                    child.family_id = self.family_id
                child.generation = self.generation + 1
                child.is_survivor = False
                child.is_descendant = self.is_survivor
                
                # Clamp

                child.resistance = max(0, child.resistance)
                child.health = max(0, child.health)
                child.speedmax = max(0, child.speedmax)
                child.agility = max(0, child.agility)
                child.punch_damage = max(0, child.punch_damage)
                child.punch_cooldown = max(0, child.punch_cooldown)
                child.shoot_cooldown = max(0, child.shoot_cooldown)
                child.gestation = max(0, child.gestation)
                child.bullet_quality = max(0, child.bullet_quality)
                child.vision = max(40, child.vision)
                child.family_id = max(0, min(255, child.family_id))
        
        
                entities.append(child)
                all_entities_epoch.append(child)
            self.reproduce_cooldown = 300
            self.punch_cooldown = PUNCH_COOLDOWN
            self.shoot_cooldown = SHOOTING_COOLDOWN

        # Detect bullet hits

        for bullet in bullets[:]:
            bulletdist = np.hypot(bullet.x - self.x, bullet.y - self.y)
            if bulletdist < 20:
                if hasattr(bullet, 'shooter') and bullet.shooter is self:
                    continue  # Ignore self-hits
                elif hasattr(bullet, 'shooter'):
                    bullet.shooter.reward += 0.25
                self.health -= DAMAGE_BULLET - self.resistance
                self.reward -= 0.25
                self.vx += ((bullet.x - self.x)/bulletdist) * -10
                self.vy += ((bullet.y - self.y)/bulletdist) * -10

                if self.health <= 0:
                    punch_effects.append(PunchEffect(self.x, self.y, RED))
                    # Find who shot the bullet
                    if hasattr(bullet, 'shooter'):
                        bullet.shooter.hunger = max(0, bullet.shooter.hunger - 80)
                        bullet.shooter.reward += 2.5
                    self.alive = False
                bullets.remove(bullet)

        if self.health <= 0:
            self.alive = False

        self.hunger += (HUNGER_RATE)
        if self.speed < 2:
            self.hunger += 0.25
        if not punch and not shoot:
            self.hunger += 0.05
        if self.hunger >= MAX_HUNGER:
            self.health -= self.hunger - MAX_HUNGER
            self.reward -= 0.1

        # Regenerate

        self.regen = False
        if output[8] > 0.5:
            self.hunger += 0.5
            self.health += self.regen_factor
            self.regen = True

        # Prevent entities from overlapping
        for other in entities:
            if other is self or not other.alive:
                continue
            if self.check_collision(other):
                dx = self.x - other.x
                dy = self.y - other.y
                dist = np.hypot(dx, dy) + 1e-5  # avoid division by zero
                overlap = 20 - dist
                if overlap > 0:
                    self.x += (dx / dist) * overlap / 2
                    self.y += (dy / dist) * overlap / 2
                    other.x -= (dx / dist) * overlap / 2
                    other.y -= (dy / dist) * overlap / 2

        # Keep entities on screen

        self.x = self.x % WIDTH
        self.y = self.y % HEIGHT

        self.color = (
        max(0, min(255, int(self.health*2.55))),
        max(0, min(255, int(self.speed*2.55))),
        self.family_id
        )

    def draw(self, screen, selected_entity):
        base_color = self.color

        if self is selected_entity:
            # Flash effect using time
            t = pygame.time.get_ticks() / 200  # speed of flashing
            pulse = (np.sin(t) + 1) / 2  # 0 → 1

            # Blend between white and gold
            white = np.array([255, 255, 255])
            gold = np.array([255, 215, 0])

            flash_color = white * pulse + gold * (1 - pulse)
            color = tuple(flash_color.astype(int))
        elif self.dash_cooldown > 0:
            color = (
                min(255, base_color[0] + 100),
                min(255, base_color[1] + 100),
                min(255, base_color[2] + 100)
            )
        else:
            color = base_color

        pygame.draw.rect(screen, color, (int(self.x), int(self.y), 20, 20))

    def isdead(self):
        return not self.alive

    

        
# Initialize Pygame
pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)

# Make entities
entities = [Entity(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(25)]
all_entities_epoch = []
all_entities_epoch.extend(entities)
rank_map = {}

extinction_triggered = False
extinction_timer = 0
selected_entity = None

while True:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()

            for entity in entities:
                if entity.alive:
                    if abs(entity.x - mx) < 30 and abs(entity.y - my) < 30:
                        selected_entity = entity
                        break
                    else:
                        selected_entity = None

    mouse_x, mouse_y = pygame.mouse.get_pos()

    screen.fill(BLACK)

    # Update and draw entities

    for entity in entities[:]:
        entity.update(entities)
        entity.draw(screen, selected_entity)

        # Name rendering
                # Distance from mouse
        dx = entity.x + 10 - mouse_x
        dy = entity.y + 10 - mouse_y
        distance = (dx**2 + dy**2)**0.5

        NAME_RADIUS = 100  # how far the mouse can be to show the name

        if distance <= NAME_RADIUS:
            # Fade factor: closer = more opaque, farther = more transparent
            alpha = max(0, min(255, int(255 * (1 - (distance / NAME_RADIUS)))))

            font = pygame.font.SysFont(None, 24)
            if entity.is_survivor == True:
                text_surface = font.render(f"{entity.id}, {rank_map.get(entity, -1) + 1}*", True, (255, 215, 0))
            elif entity.is_descendant == True:
                text_surface = font.render(f"{entity.id}, {rank_map.get(entity, -1) + 1}", True, (255, 50, 255))
            else:
                text_surface = font.render(f"{entity.id}, {rank_map.get(entity, -1) + 1}", True, (255, 255, 255))
    
            # Create a surface that supports alpha
            text_surface_alpha = text_surface.convert_alpha()
            text_surface_alpha.set_alpha(alpha)
    
            screen.blit(text_surface_alpha, (entity.x - 20, entity.y - 20))

    for bullet in bullets[:]:
        bullet.update()
        pygame.draw.circle(screen, GREEN, (int(bullet.x), int(bullet.y)), 5)
        if bullet.bounce <= 0:
            bullets.remove(bullet)

    for effect in punch_effects[:]:
        effect.update()
        effect.draw(screen)
        if effect.duration <= 0:
            punch_effects.remove(effect)

    # Choose survivors 
    top_survivors_global = []
    mean_epoch_reward = 0

    if all_entities_epoch:
        entities_sorted = sorted(all_entities_epoch, key=lambda e: e.reward, reverse=True)
        top_survivors_global = entities_sorted[:20]
        rank_map = {e: i for i, e in enumerate(entities_sorted)}

    for entity in all_entities_epoch:
        mean_epoch_reward += entity.reward
    mean_epoch_reward = mean_epoch_reward / len(all_entities_epoch)
    
    # Find the king
    best_organism = None
    if all_entities_epoch:
        best_organism = max(all_entities_epoch, key=lambda e: e.reward)

    if selected_entity and selected_entity.alive:
        font = pygame.font.SysFont(None, 24)

        panel_x, panel_y = 10, 10
        line_height = 20

        bio_lines = [
            f"ID: {selected_entity.id}",
            f"Age: {int(selected_entity.age/60)}",
            f"Health: {int(selected_entity.health)}",
            f"Hunger: {int(selected_entity.hunger)}",
            f"Layer 1 Neurons: {selected_entity.l1_size}",
            f"Layer 2 Neurons: {selected_entity.l2_size}",
            f"Speed: {round(selected_entity.speed, 2)}",
            f"Agility: {round(selected_entity.agility, 2)}",
            f"Aggro: {round(selected_entity.aggro_bias, 2)}",
            f"Family: {selected_entity.family_id}",
            f"Vision: {int(selected_entity.vision)}",
            f"Generation: {selected_entity.generation}",
            f"Survivor: {selected_entity.is_survivor}",
            f"Reward Score: {selected_entity.reward}",
            f"Epoch Rank: {rank_map.get(selected_entity, -1)}"
        ]

        # Background box
        pygame.draw.rect(screen, (30, 30, 30), (panel_x-5, panel_y-5, 250, len(bio_lines)*line_height + 10))

        for i, line in enumerate(bio_lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (panel_x, panel_y + i * line_height))

    # Remove dead entities after updates
    entities = [e for e in entities if e.alive]

    if len(entities) < 1:
        # Begin new epoch
        top10survivors = top_survivors_global[:10]
        top20survivors = top_survivors_global[:20]
        all_entities_epoch = []
        entities = []

        # Add survivors
        for s in top10survivors:
            s.is_survivor = True
            s.x = random.randint(0, WIDTH)
            s.y = random.randint(0, HEIGHT)
            s.health = 100
            s.hunger = 0
            s.age = 0
            s.alive = True
            s.reward = 5.0 # higher starting reward for survivors
            entities.append(s)

        # Add mutated offspring of survivors

        max_children = 5
        min_children = 1
        total_survivors = len(top20survivors)

        for s in top20survivors:
            num_children = random.randint(1, 2)

            rank = rank_map[s]
            # Linear formula to decide who gets how many children
            num_children = min_children + int((total_survivors - 1 - rank) * (max_children - min_children) / (total_survivors - 1))

            for _ in range(num_children):
                child = Entity(
                s.x + random.randint(-50, 50),
                s.y + random.randint(-50, 50)
                )

                # Copy and mutate brain

                for i in range(len(s.brain.weights)):
                    child.brain.weights[i] = s.brain.weights[i] + np.random.normal(0, 0.05, s.brain.weights[i].shape)
                    child.brain.biases[i] = s.brain.biases[i] + np.random.normal(0, 0.05, s.brain.biases[i].shape)

                # Mutate traits (lighter mutations than reproduction for stability)
                child.speedmax = max(1, s.speedmax + random.uniform(-2, 2))
                child.aggro_bias = s.aggro_bias + random.uniform(-0.2, 0.2)
                child.punch_damage = max(0, s.punch_damage + random.uniform(-2, 2))
                child.vision = max(40, s.vision + random.uniform(-20, 20))
                child.regen_factor = max(0, s.regen_factor + random.uniform(-0.1, 0.1))

                # Mutate brain topology

                add_chance = 0.01 
                remove_chance = 0.005 # less chance of a mutation than normal reproduction

                if random.uniform(0, 1) <= add_chance:
                    child.l1_size += 1
                if random.uniform(0, 1) <= remove_chance and child.l2_size > 0:
                    child.l1_size -= 1

                if random.uniform(0, 1) <= add_chance:
                    child.l2_size += 1
                if random.uniform(0, 1) <= remove_chance and child.l2_size > 0:
                    child.l2_size -= 1

                # Identity
                child.id = mutate_id(s.id)
                child.family_id = s.family_id
                child.generation = s.generation + 1
                child.is_survivor = False  # important: only originals are "true survivors"
                child.is_descendant = True
                child.reward = 3.0 # smaller child bonus

                entities.append(child)
                all_entities_epoch.append(child)

        # Add new random entities
        for i in range(10):
            new_entity = Entity(random.randint(0, WIDTH), random.randint(0, HEIGHT))
            entities.append(new_entity)
            all_entities_epoch.append(new_entity)

        # Update epoch
        epoch += 1
        epoch_name = mutate_id(epoch_name[:-4])
        for i in range(random.randint(0,3)):
            epoch_name = mutate_id(epoch_name)
        epoch_name += 'CENE'
        all_entities_epoch.extend(entities)

    if len(entities) > 199 and not extinction_triggered:
        extinction_triggered = True
        extinction_timer = 1
        # Perfectly balanced, as all things should be. Or not.
        kill_ratio = 0.99
        num_to_kill = int(len(entities) * kill_ratio)
        to_kill = random.sample(entities, num_to_kill)
        for entity in to_kill:
            entity.alive = False

        # Spawn a few random entities
        for i in range(random.randint(10, 20)):
            new_entity = Entity(random.randint(0, WIDTH), random.randint(0, HEIGHT))
            entities.append(new_entity)
            all_entities_epoch.append(new_entity)
            
    if extinction_triggered:

        flash_surface = pygame.Surface((WIDTH, HEIGHT))
        flash_surface.fill((255, 255, 255))

        flash_alpha = max(0, 255 - (extinction_timer * 4))  # adjust speed: 4 per frame
        flash_surface.set_alpha(flash_alpha)
        screen.blit(flash_surface, (0, 0))

        font = pygame.font.SysFont(None, 100)
        text_surface = font.render('MASS EXTINCTION!', True, (255, 0, 0))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(text_surface, text_rect)

        extinction_timer += 1
        if extinction_timer >= 60:  # show for ~1 second
            extinction_timer = 0
            extinction_triggered = False

    font = pygame.font.SysFont(None, 48)  # default font, size 48
    text_surface = font.render(f"{epoch_name}({epoch + 1})", True, (255, 255, 255))
    screen.blit(text_surface, (730, 10))
    font = pygame.font.SysFont(None, 24)  # default font, size 24
    text_surface = font.render(f"Epoch Score: {round(mean_epoch_reward, 10)}", True, (255, 255, 255))
    screen.blit(text_surface, (740, 45))
    font = pygame.font.SysFont(None, 24)  # default font, size 24
    text_surface = font.render(f"Best Organism: {best_organism.id}", True, (255, 255, 255))
    screen.blit(text_surface, (740, 65))
    
    pygame.display.flip()