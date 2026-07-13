import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os

class WarehouseSceneGenerator:
    """库房场景点云数据生成器"""
    
    def __init__(self, warehouse_size=(50, 40, 5)):
        """
        初始化库房场景
        warehouse_size: (length, width, height) 库房尺寸
        """
        self.length, self.width, self.height = warehouse_size
        self.points = []
        self.colors = []
        
    def add_points(self, points, color):
        """添加点云和颜色"""
        if len(points) > 0:
            self.points.append(points)
            colors = np.full((len(points), 3), color)
            self.colors.append(colors)
    
    def generate_wall_points(self, x_range, y_range, z_range, density=1000):
        """生成墙体点云"""
        # 随机生成墙体表面的点
        x = np.random.uniform(x_range[0], x_range[1], density)
        y = np.random.uniform(y_range[0], y_range[1], density)
        z = np.random.uniform(z_range[0], z_range[1], density)
        return np.stack([x, y, z], axis=1)
    
    def generate_box_points(self, center, size, density=2000, noise=0.02):
        """
        生成长方体点云
        center: (x, y, z) 中心点
        size: (length, width, height) 尺寸
        """
        cx, cy, cz = center
        lx, ly, lz = size
        
        # 生成长方体表面的点
        points = []
        
        # 前后面
        for _ in range(density // 6):
            x = np.random.uniform(cx - lx/2, cx + lx/2, 1)[0]
            y = np.random.uniform(cy - ly/2, cy + ly/2, 1)[0]
            z = np.random.uniform(cz - lz/2, cz + lz/2, 1)[0]
            # 前面
            points.append([x, cy - ly/2, z])
            # 后面
            points.append([x, cy + ly/2, z])
        
        # 左右面
        for _ in range(density // 6):
            x = np.random.uniform(cx - lx/2, cx + lx/2, 1)[0]
            y = np.random.uniform(cy - ly/2, cy + ly/2, 1)[0]
            z = np.random.uniform(cz - lz/2, cz + lz/2, 1)[0]
            # 左面
            points.append([cx - lx/2, y, z])
            # 右面
            points.append([cx + lx/2, y, z])
        
        # 上下面
        for _ in range(density // 3):
            x = np.random.uniform(cx - lx/2, cx + lx/2, 1)[0]
            y = np.random.uniform(cy - ly/2, cy + ly/2, 1)[0]
            z = np.random.uniform(cz - lz/2, cz + lz/2, 1)[0]
            # 下面
            points.append([x, y, cz - lz/2])
            # 上面
            points.append([x, y, cz + lz/2])
        
        return np.array(points)
    
    def generate_cylinder_points(self, center, radius, height, density=1500):
        """生成圆柱体点云（用于货架支柱）"""
        cx, cy, cz = center
        points = []
        
        # 侧面
        theta = np.random.uniform(0, 2*np.pi, density // 2)
        z = np.random.uniform(cz - height/2, cz + height/2, density // 2)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        points.extend(np.stack([x, y, z], axis=1))
        
        # 上下圆面
        for _ in range(density // 4):
            r = np.random.uniform(0, radius, 1)[0]
            theta = np.random.uniform(0, 2*np.pi, 1)[0]
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            # 下面
            points.append([x, y, cz - height/2])
            # 上面
            points.append([x, y, cz + height/2])
        
        return np.array(points)
    
    def generate_forklift_points(self, center, rotation_angle=0, forklift_type='standard'):
        """
        生成叉车点云
        forklift_type: 'standard'(标准叉车), 'compact'(紧凑型), 'reach'(伸缩臂)
        """
        cx, cy, cz = center
        points = []
        
        if forklift_type == 'standard':
            # 标准叉车：车身 + 叉子
            # 车身 (长x宽x高: 3x1.5x2)
            body_points = self.generate_box_points((cx, cy, cz + 1), (3, 1.5, 2), density=1500)
            points.extend(body_points)
            
            # 前叉 (两个平行的叉子)
            fork_length = 1.2
            fork_width = 0.15
            fork_height = 0.1
            # 左叉
            left_fork = self.generate_box_points((cx + 1.2, cy - 0.4, cz + 0.5), 
                                                 (fork_length, fork_width, fork_height), density=800)
            points.extend(left_fork)
            # 右叉
            right_fork = self.generate_box_points((cx + 1.2, cy + 0.4, cz + 0.5), 
                                                  (fork_length, fork_width, fork_height), density=800)
            points.extend(right_fork)
            
            # 支柱
            mast_points = self.generate_cylinder_points((cx - 0.5, cy, cz + 1.5), 0.15, 1.5, density=600)
            points.extend(mast_points)
            
        elif forklift_type == 'compact':
            # 紧凑型叉车：更小的车身
            body_points = self.generate_box_points((cx, cy, cz + 0.8), (2.5, 1.2, 1.6), density=1200)
            points.extend(body_points)
            
            fork_length = 1.0
            fork_width = 0.12
            fork_height = 0.08
            left_fork = self.generate_box_points((cx + 1.0, cy - 0.35, cz + 0.4), 
                                                 (fork_length, fork_width, fork_height), density=600)
            points.extend(left_fork)
            right_fork = self.generate_box_points((cx + 1.0, cy + 0.35, cz + 0.4), 
                                                  (fork_length, fork_width, fork_height), density=600)
            points.extend(right_fork)
            
            mast_points = self.generate_cylinder_points((cx - 0.4, cy, cz + 1.2), 0.12, 1.2, density=500)
            points.extend(mast_points)
            
        elif forklift_type == 'reach':
            # 伸缩臂叉车：有伸缩机制
            body_points = self.generate_box_points((cx, cy, cz + 1.2), (3.2, 1.6, 2.2), density=1600)
            points.extend(body_points)
            
            fork_length = 1.5
            fork_width = 0.15
            fork_height = 0.1
            left_fork = self.generate_box_points((cx + 1.5, cy - 0.45, cz + 0.6), 
                                                 (fork_length, fork_width, fork_height), density=900)
            points.extend(left_fork)
            right_fork = self.generate_box_points((cx + 1.5, cy + 0.45, cz + 0.6), 
                                                  (fork_length, fork_width, fork_height), density=900)
            points.extend(right_fork)
            
            # 更长的支柱
            mast_points = self.generate_cylinder_points((cx - 0.6, cy, cz + 1.8), 0.18, 2.0, density=700)
            points.extend(mast_points)
        
        # 应用旋转
        if rotation_angle != 0:
            points = np.array(points)
            rotation_matrix = R.from_euler('z', rotation_angle, degrees=True).as_matrix()
            # 绕中心旋转
            points = points - np.array([cx, cy, cz])
            points = points @ rotation_matrix.T
            points = points + np.array([cx, cy, cz])
        
        return np.array(points)
    
    def generate_shelf_points(self, center, shelf_type='small'):
        """
        生成货架点云
        shelf_type: 'small'(1m), 'medium'(2m), 'large'(3m)
        """
        cx, cy, cz = center
        points = []
        
        if shelf_type == 'small':
            # 小货架：1m高
            height = 1.0
            width = 1.2
            depth = 0.6
        elif shelf_type == 'medium':
            # 中型货架：2m高
            height = 2.0
            width = 1.5
            depth = 0.8
        else:  # large
            # 大型货架：3m高
            height = 3.0
            width = 1.5
            depth = 0.8
        
        # 四个支柱
        pillar_radius = 0.08
        pillar_positions = [
            (cx - width/2 + 0.1, cy - depth/2 + 0.1),
            (cx + width/2 - 0.1, cy - depth/2 + 0.1),
            (cx - width/2 + 0.1, cy + depth/2 - 0.1),
            (cx + width/2 - 0.1, cy + depth/2 - 0.1),
        ]
        
        for px, py in pillar_positions:
            pillar = self.generate_cylinder_points((px, py, cz + height/2), pillar_radius, height, density=400)
            points.extend(pillar)
        
        # 横梁（多层）
        num_shelves = int(height / 0.5) + 1
        for i in range(num_shelves):
            z = cz + i * 0.5
            # 前后横梁
            for y_offset in [-depth/2 + 0.1, depth/2 - 0.1]:
                beam = self.generate_box_points((cx, cy + y_offset, z), 
                                               (width - 0.2, 0.1, 0.08), density=300)
                points.extend(beam)
            # 左右横梁
            for x_offset in [-width/2 + 0.1, width/2 - 0.1]:
                beam = self.generate_box_points((cx + x_offset, cy, z), 
                                               (0.1, depth - 0.2, 0.08), density=300)
                points.extend(beam)
        
        return np.array(points)
    
    def generate_workbench_points(self, center):
        """生成工作台点云（1m高）"""
        cx, cy, cz = center
        points = []
        
        # 工作台面 (1.5m x 0.8m x 0.05m)
        tabletop = self.generate_box_points((cx, cy, cz + 0.95), (1.5, 0.8, 0.05), density=800)
        points.extend(tabletop)
        
        # 四个腿 (0.1m x 0.1m x 0.95m)
        leg_positions = [
            (cx - 0.7, cy - 0.35),
            (cx + 0.7, cy - 0.35),
            (cx - 0.7, cy + 0.35),
            (cx + 0.7, cy + 0.35),
        ]
        
        for lx, ly in leg_positions:
            leg = self.generate_box_points((lx, ly, cz + 0.475), (0.1, 0.1, 0.95), density=400)
            points.extend(leg)
        
        return np.array(points)
    
    def generate_ground_points(self, density=5000):
        """生成地面点云"""
        x = np.random.uniform(-self.length/2, self.length/2, density)
        y = np.random.uniform(-self.width/2, self.width/2, density)
        z = np.zeros(density)
        return np.stack([x, y, z], axis=1)
    
    def generate_complete_scene(self):
        """生成完整的库房场景"""
        
        # 1. 地面
        ground = self.generate_ground_points(density=8000)
        self.add_points(ground, [0.5, 0.5, 0.5])  # 灰色
        
        # 2. 四面墙
        wall_thickness = 0.2
        wall_height = self.height
        
        # 前墙
        front_wall = self.generate_wall_points(
            (-self.length/2, self.length/2),
            (-self.width/2, -self.width/2 + wall_thickness),
            (0, wall_height),
            density=3000
        )
        self.add_points(front_wall, [0.7, 0.7, 0.7])  # 浅灰色
        
        # 后墙
        back_wall = self.generate_wall_points(
            (-self.length/2, self.length/2),
            (self.width/2 - wall_thickness, self.width/2),
            (0, wall_height),
            density=3000
        )
        self.add_points(back_wall, [0.7, 0.7, 0.7])
        
        # 左墙
        left_wall = self.generate_wall_points(
            (-self.length/2, -self.length/2 + wall_thickness),
            (-self.width/2, self.width/2),
            (0, wall_height),
            density=3000
        )
        self.add_points(left_wall, [0.7, 0.7, 0.7])
        
        # 右墙
        right_wall = self.generate_wall_points(
            (self.length/2 - wall_thickness, self.length/2),
            (-self.width/2, self.width/2),
            (0, wall_height),
            density=3000
        )
        self.add_points(right_wall, [0.7, 0.7, 0.7])
        
        # 3. 两个工作台（1m高）
        workbench1 = self.generate_workbench_points((-15, -12, 0))
        self.add_points(workbench1, [0.8, 0.6, 0.4])  # 棕色
        
        workbench2 = self.generate_workbench_points((15, 12, 0))
        self.add_points(workbench2, [0.8, 0.6, 0.4])
        
        # 4. 两个中型货架（2m高）
        shelf_medium1 = self.generate_shelf_points((-10, 0, 0), shelf_type='medium')
        self.add_points(shelf_medium1, [0.6, 0.6, 0.6])  # 深灰色
        
        shelf_medium2 = self.generate_shelf_points((10, 0, 0), shelf_type='medium')
        self.add_points(shelf_medium2, [0.6, 0.6, 0.6])
        
        # 5. 两个大型货架（3m高，由两个中型货架拼起来）
        shelf_large1 = self.generate_shelf_points((-5, -15, 0), shelf_type='large')
        self.add_points(shelf_large1, [0.5, 0.5, 0.5])  # 更深的灰色
        
        shelf_large2 = self.generate_shelf_points((5, 15, 0), shelf_type='large')
        self.add_points(shelf_large2, [0.5, 0.5, 0.5])
        
        # 6. 多种叉车
        # 标准叉车
        forklift_standard = self.generate_forklift_points((-20, -8, 0), rotation_angle=0, forklift_type='standard')
        self.add_points(forklift_standard, [1.0, 0.0, 0.0])  # 红色
        
        # 紧凑型叉车
        forklift_compact = self.generate_forklift_points((20, 8, 0), rotation_angle=45, forklift_type='compact')
        self.add_points(forklift_compact, [0.0, 1.0, 0.0])  # 绿色
        
        # 伸缩臂叉车
        forklift_reach = self.generate_forklift_points((0, -18, 0), rotation_angle=90, forklift_type='reach')
        self.add_points(forklift_reach, [0.0, 0.0, 1.0])  # 蓝色
        
        # 另一个标准叉车
        forklift_standard2 = self.generate_forklift_points((0, 18, 0), rotation_angle=180, forklift_type='standard')
        self.add_points(forklift_standard2, [1.0, 1.0, 0.0])  # 黄色
    
    def save_as_ply(self, filename):
        """保存为PLY格式"""
        if len(self.points) == 0:
            print("没有点云数据")
            return
        
        points = np.vstack(self.points)
        colors = np.vstack(self.colors)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"点云已保存为 {filename}")
        print(f"总点数: {len(points)}")
    
    def save_as_kitti(self, filename):
        """保存为KITTI格式（.bin文件）"""
        if len(self.points) == 0:
            print("没有点云数据")
            return
        
        points = np.vstack(self.points)
        
        # KITTI格式：每个点是 (x, y, z, intensity)
        # 这里用颜色的平均值作为intensity
        colors = np.vstack(self.colors)
        intensity = np.mean(colors, axis=1, keepdims=True)
        
        # 组合为 (x, y, z, intensity)
        kitti_points = np.hstack([points, intensity])
        
        # 保存为二进制文件
        kitti_points.astype(np.float32).tofile(filename)
        print(f"KITTI格式点云已保存为 {filename}")
        print(f"总点数: {len(kitti_points)}")
        
        # 同时保存为文本格式便于查看
        txt_filename = filename.replace('.bin', '.txt')
        np.savetxt(txt_filename, kitti_points, fmt='%.6f', delimiter=' ')
        print(f"文本格式已保存为 {txt_filename}")


if __name__ == '__main__':
    # 创建输出目录
    output_dir = 'demo_data/warehouse_scene'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建库房场景生成器
    generator = WarehouseSceneGenerator(warehouse_size=(50, 40, 5))
    
    # 生成完整场景
    generator.generate_complete_scene()
    
    # 保存为PLY格式
    generator.save_as_ply(os.path.join(output_dir, 'warehouse_scene.ply'))
    
    # 保存为KITTI格式
    generator.save_as_kitti(os.path.join(output_dir, 'warehouse_scene.bin'))
    
    print("\n库房场景生成完成！")
    print("包含内容：")
    print("- 库房四周的墙体")
    print("- 2个工作台（1m高）")
    print("- 2个中型货架（2m高）")
    print("- 2个大型货架（3m高）")
    print("- 4种不同类型的叉车（标准、紧凑、伸缩臂）")
    print(f"\n数据已保存到: {output_dir}")