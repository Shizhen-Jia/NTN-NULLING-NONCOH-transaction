"""Independent scene variant: one dominance-validated TN per rectangle-sector."""
import json
from pathlib import Path
import numpy as np
from tn_sector_drop import SectorTNSampler, channel_dominance
from ntn_music_detection import collapse_cir_to_narrowband
from multipath_support import resolve_propagation_options
import sionna
from satellite_projection import satellite_projection
from sionna.rt import PathSolver, PlanarArray
from sionnautils.miutils import CoverageMapPlanner

# Importing this module registers the "vsat_dish" antenna pattern.
import vsat_dish_3gpp  # noqa: F401

class SceneConfigSionna:
    def __init__(self, scene):
        """
        scene : A sionna.rt.Scene object with loaded geometry (XML or otherwise).
        """
        self.scene = scene
        
        self.fc = 7e9

        # Some default parameters you can modify:
        self.grid_size = 1.0
        self.nbs = None               # Number of gNB/base stations
        self.nsect = None              # Usually 3, for sector-based coverage
        # self.BS_height_above_roof = 35  # For base station on building
        # self.BS_height_above_ground = 45
        self.BS_height_above_roof = 35  # For base station on building
        self.BS_height_above_ground = 35
        # self.tn_height_above_roof = 1.2  # For base station on building
        self.tn_height_above_roof = -1.5  # For base station on building
        self.tn_height_above_ground = 1.8
        self.ntn_height_above_roof = 1.2  # For base station on building
        self.ntn_height_above_ground = 1.88
        self.sat_distance = 550e3   # Satellite distance from region (m)

        # Coverage map placeholders
        self.cm = None
        self.L_NS = None
        self.W_WE = None
        self.bbox = None
        self.extent = None
        self.point_type = None
        self.paths_tn = None
        self.paths_ntn = None

        # Position arrays
        self.tx_pos = None
        self.rx_ntn_pos = None
        self.tn_pos = None
        self.ntn_look_pos = None

        # Path results
        self.a_tn = None
        self.tau_tn = None
        self.a_ntn = None
        self.tau_ntn = None
        self.ntn_rx = None
        
        self.toff = None
        self.h_tf = None
        self.tn_bs_index = None
        self.tn_sector_index = None
        self.tx_bs_index = None
        self.tx_sector_index = None
        self.tx_orientation_rad = None
        self.tx_name_list = None
        # Default BS sector orientation controls (can be overwritten from notebook)
        self.tx_sector_yaw_offset_rad = 0.0
        self.tx_sector_pitch_rad = 0.174533  # Positive Sionna pitch tilts +x boresight downward
        self.tx_sector_roll_rad = 0.0

    def build_coverage_map(self, grid_size=None, show_xy=False, plot=False):
        """
        Build coverage map, compute bbox/extent/point_type.
        Optional: print x/y ranges and plot building/outdoor map.
        """
        if grid_size is not None:
            self.grid_size = grid_size

        self.cm = CoverageMapPlanner(self.scene._scene, grid_size=self.grid_size)
        self.cm.set_grid()
        self.cm.compute_grid_attributes()
        self.hm = None

        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        self.W_WE = x_max - x_min   # width (East-West)
        self.L_NS = y_max - y_min   # length (North-South)
        self.bbox = [-self.W_WE/2, self.W_WE/2, -self.L_NS/2, self.L_NS/2]
        # self.extent = [self.cm.x[0], self.cm.x[-1], self.cm.y[0], self.cm.y[-1]]
        self.extent = [x_min, x_max, y_min, y_max]

        building = (self.cm.bldg_grid)
        self.point_type = np.where(building, 2, 1).astype(np.int8)
        self.point_type = np.flipud(self.point_type)
        # outdoor = (self.cm.bldg_grid == False)
        # self.point_type = outdoor + 2 * building
        # self.point_type = self.point_type.astype(int)
        # self.point_type = np.flipud(self.point_type)

        if show_xy:
            print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
            print(f"y range: [{y_min:.3f}, {y_max:.3f}]")

        if plot:
            self._plot_grid()

    def _plot_grid(self, show_bs=False, show_tn=False, show_ntn=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        colors = ['lightgray', 'brown']
        cmap = ListedColormap(colors)
        plt.figure()
        plt.imshow(self.point_type, cmap=cmap, interpolation='nearest', extent=self.extent)
        if show_bs and self.tx_pos is not None:
            bs_marker = (3, 0, -30)  # triangle rotated 30 degrees clockwise
            plt.scatter(self.tx_pos[:, 0], self.tx_pos[:, 1], c="red", s=60, marker=bs_marker, label="BS")
        if show_tn and self.tn_pos is not None:
            plt.scatter(self.tn_pos[:, 0], self.tn_pos[:, 1], c="green", s=20, marker="o", label="TN")
        if show_ntn and self.rx_ntn_pos is not None:
            plt.scatter(self.rx_ntn_pos[:, 0], self.rx_ntn_pos[:, 1], c="blue", s=20, marker="x", label="NTN")
        has_overlays = (
            (show_bs and self.tx_pos is not None)
            or (show_tn and self.tn_pos is not None)
            or (show_ntn and self.rx_ntn_pos is not None)
        )
        if has_overlays:
            plt.legend(loc="upper right")
        plt.xlabel("x")
        plt.ylabel("y")
        if has_overlays:
            plt.title("Spatial Distribution of TN/NTN UEs and BSs")
        else:
            plt.title("Building/Outdoor Map")
        plt.show()

    def _snap_to_grid(self, x, y, height_roof=None, height_ground=None):
        x = np.asarray(x)
        y = np.asarray(y)
        ix = np.searchsorted(self.cm.x, x)
        iy = np.searchsorted(self.cm.y, y)
        ix = np.clip(ix, 0, len(self.cm.x) - 1)
        iy = np.clip(iy, 0, len(self.cm.y) - 1)

        if height_roof is None:
            height_roof = self.BS_height_above_roof
        if height_ground is None:
            height_ground = self.BS_height_above_ground

        xg = self.cm.x[ix]
        yg = self.cm.y[iy]
        z = np.where(
            self.cm.bldg_grid[iy, ix],
            self.cm.zmax_grid[iy, ix] + height_roof,
            self.cm.zmin_grid[iy, ix] + height_ground
        )
        return xg, yg, z

    def _positions_from_indices(self, indices, height_roof, height_ground):
        x = self.cm.x[indices[:, 1]]
        y = self.cm.y[indices[:, 0]]
        z = np.where(
            self.cm.bldg_grid[indices[:, 0], indices[:, 1]],
            self.cm.zmax_grid[indices[:, 0], indices[:, 1]] + height_roof,
            self.cm.zmin_grid[indices[:, 0], indices[:, 1]] + height_ground
        )
        return np.column_stack((x, y, z))

    def compute_positions(self,
                        ntn_rx,
                        tn_rx,
                        azimuth,
                        elevation,
                        centerBS=True,
                        bs_dist_max=1000,
                        bs_boundary=0.0,
                        bs_layout="random",
                        bs_grid=None,
                        nbs=None,
                        tn_distance=1500.0,
                        tn_building_ratio="sector",
                        ntn_building_ratio=None,
                        tn_outdoor_probability=1.0,
                        tn_association_margin_db=3.0,
                        tn_min_channel_norm=0.0,
                        tn_max_attempts=256,
                        tn_candidates_per_batch=16,
                        tn_sector_yaw_offset_rad=0.0,
                        tn_drop_output_dir=None,
                        show_xy=False,
                        plot_grid=False,
                        plot_bs=False,
                        plot_tn=False,
                        plot_ntn=False):
        """
        1) Build coverage map and determine bounding box
        2) Select random positions for TX, TN/NTN receivers
        3) Optionally place TX at (x=0, y=0) if centerBS=True
        4) Sample one TN in each rectangle-sector region
        5) Compute random satellite direction & project
        """

        if centerBS or tuple(bs_grid or ()) != (2, 2) or tn_rx != 12:
            raise ValueError("Sector drop requires centerBS=False, bs_grid=(2, 2), tn_rx=12.")
        if (not np.isfinite(tn_association_margin_db) or tn_association_margin_db < 0
                or not np.isfinite(tn_min_channel_norm) or tn_min_channel_norm < 0):
            raise ValueError("Association margin and minimum channel norm must be finite and nonnegative.")
        if (isinstance(tn_candidates_per_batch, bool) or tn_candidates_per_batch < 1
                or int(tn_candidates_per_batch) != tn_candidates_per_batch):
            raise ValueError("tn_candidates_per_batch must be a positive integer.")

        # 1) Create/reuse coverage map
        self.ntn_rx = ntn_rx
        if self.cm is None or self.point_type is None or self.extent is None:
            self.build_coverage_map(show_xy=show_xy, plot=False)
        else:
            if show_xy:
                x_min, x_max = self.cm.x[0], self.cm.x[-1]
                y_min, y_max = self.cm.y[0], self.cm.y[-1]
                print(f"x range: [{x_min:.3f}, {x_max:.3f}]")
                print(f"y range: [{y_min:.3f}, {y_max:.3f}]")
        x_min, x_max = self.cm.x[0], self.cm.x[-1]
        y_min, y_max = self.cm.y[0], self.cm.y[-1]

        # 2) Place a single gNB TX position on building roof
        # locations_building = np.argwhere(self.cm.bldg_grid & self.cm.in_region)        
        locations_building = np.argwhere(self.cm.bldg_grid)
        locations_outdoor = np.argwhere(~self.cm.bldg_grid)
        
        # if len(locations_building) < self.nbs:
        #     raise ValueError("Not enough building points to place the TX.")

        # If centerBS=True, force a single BS at (0,0) regardless of bs_grid
        if centerBS:
            self.nbs = 1
        elif bs_grid is not None:
            nx, ny = bs_grid
            if nx <= 0 or ny <= 0:
                raise ValueError("bs_grid must be positive, e.g. (2, 2).")
            self.nbs = int(nx * ny)
        elif nbs is not None:
            self.nbs = int(nbs)
        elif self.nbs is None:
            raise ValueError("nbs is required when bs_grid is not set.")

        # If centerBS = True and only 1 BS, force TX at (0, 0).
        if centerBS and self.nbs == 1:
            tx_x, tx_y, tx_z = self._snap_to_grid(np.array([0.0]), np.array([0.0]))
        elif bs_grid is not None:
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            y_start = y_min + bs_boundary
            y_end = y_max - bs_boundary
            if x_end <= x_start or y_end <= y_start:
                raise ValueError("bs_boundary too large: no valid x/y range.")
            xs = np.linspace(x_start, x_end, nx)
            ys = np.linspace(y_start, y_end, ny)
            xx, yy = np.meshgrid(xs, ys)
            tx_x = xx.ravel()
            tx_y = yy.ravel()
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        elif bs_layout == "line":
            # Place BSs evenly on x-axis within boundary, y=0
            x_start = x_min + bs_boundary
            x_end = x_max - bs_boundary
            if x_end <= x_start:
                raise ValueError("bs_boundary too large: no valid x range.")
            tx_x = np.linspace(x_start, x_end, self.nbs)
            tx_y = np.zeros_like(tx_x)
            tx_x, tx_y, tx_z = self._snap_to_grid(tx_x, tx_y)
        else:
            x_limit_min = x_min + bs_dist_max
            x_limit_max = x_max - bs_dist_max
            y_limit_min = y_min + bs_dist_max
            y_limit_max = y_max - bs_dist_max
            x_coords = self.cm.x[locations_outdoor[:, 1]]
            y_coords = self.cm.y[locations_outdoor[:, 0]]
            mask = (
                (x_coords >= x_limit_min) & (x_coords <= x_limit_max) &
                (y_coords >= y_limit_min) & (y_coords <= y_limit_max)
            )
            locations_outdoor_limited = locations_outdoor[mask]

            # Prefer interior outdoor points; if too few, gracefully fall back.
            if locations_outdoor_limited.shape[0] >= self.nbs:
                tx_candidates = locations_outdoor_limited
            elif locations_outdoor.shape[0] > 0:
                tx_candidates = locations_outdoor
            elif locations_building.shape[0] > 0:
                tx_candidates = locations_building
            else:
                raise ValueError("No valid grid points available to place BS.")

            replace = tx_candidates.shape[0] < self.nbs
            tx_ind = tx_candidates[
                np.random.choice(tx_candidates.shape[0], self.nbs, replace=replace)
            ]
            tx_x = self.cm.x[tx_ind[:,1]]
            tx_y = self.cm.y[tx_ind[:,0]]
            tx_z = np.where(
                self.cm.bldg_grid[tx_ind[:, 0], tx_ind[:, 1]],
                self.cm.zmax_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_roof,
                self.cm.zmin_grid[tx_ind[:, 0], tx_ind[:, 1]] + self.BS_height_above_ground
            )
        self.tx_pos = np.column_stack((tx_x, tx_y, tx_z))


        # 3) Place NTN receivers
        if ntn_building_ratio is None:
            num_ntn_building = None
            num_ntn_outdoor = None
        else:
            num_ntn_building = int(round(ntn_building_ratio * ntn_rx))
            num_ntn_outdoor = ntn_rx - num_ntn_building

        if ntn_building_ratio is None:
            # Pure random from all points
            all_idx = np.argwhere(self.cm.bldg_grid | (~self.cm.bldg_grid))
            if ntn_rx > all_idx.shape[0]:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=True)
                ]
            else:
                candidate_indices = all_idx[
                    np.random.choice(all_idx.shape[0], ntn_rx, replace=False)
                ]
        else:
            # Random with building/outdoor ratio
            rx_ntn_building_ind = locations_building[
                np.random.choice(
                    locations_building.shape[0],
                    num_ntn_building,
                    replace=num_ntn_building > locations_building.shape[0]
                )
            ] if num_ntn_building > 0 else np.empty((0, 2), dtype=int)
            rx_ntn_outdoor_ind = locations_outdoor[
                np.random.choice(
                    locations_outdoor.shape[0],
                    num_ntn_outdoor,
                    replace=num_ntn_outdoor > locations_outdoor.shape[0]
                )
            ] if num_ntn_outdoor > 0 else np.empty((0, 2), dtype=int)
            candidate_indices = np.vstack((rx_ntn_building_ind, rx_ntn_outdoor_ind))

        def filter_indices(indices):
            # 根据 cm 坐标映射获得 x, y
            rx_ntn_x = self.cm.x[indices[:, 1]]
            rx_ntn_y = self.cm.y[indices[:, 0]]
            # 过滤条件：要求不在中心区域内
            # 比如，中心 500x500 米区域：x 和 y 同时满足 |x|<250 且 |y|<250
            mask = ~((np.abs(rx_ntn_x) < 250) & (np.abs(rx_ntn_y) < 800))
            return indices[mask]

        # 初步过滤
        filtered_indices = filter_indices(candidate_indices)

        # 如果数量不足 ntn_rx，则不断补充
        while filtered_indices.shape[0] < ntn_rx:
            # 为补充，按照比例重新采样一些候选点
            # 注意：为了防止 replace=False 时候候选点不足，可以设置 replace=True，
            # 但这可能会有重复，最后再使用 np.unique 去重
            extra_building = locations_building[
                np.random.choice(locations_building.shape[0], max(1, int(0.8 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_outdoor = locations_outdoor[
                np.random.choice(locations_outdoor.shape[0], max(1, int(0.2 * (ntn_rx - filtered_indices.shape[0])),), replace=True)
            ]
            extra_candidates = np.vstack((extra_building, extra_outdoor))
            extra_filtered = filter_indices(extra_candidates)
            # 合并，去重
            filtered_indices = np.vstack((filtered_indices, extra_filtered))
            # 去重（因为索引是整数数组，这里可以使用 np.unique ）
            filtered_indices = np.unique(filtered_indices, axis=0)

        # 最终只保留前 ntn_rx 个（如果多于 ntn_rx 个）
        if filtered_indices.shape[0] > ntn_rx:
            filtered_indices = filtered_indices[:ntn_rx]

        # 更新映射后的 x, y 坐标
        rx_ntn_x = self.cm.x[filtered_indices[:, 1]]
        rx_ntn_y = self.cm.y[filtered_indices[:, 0]]
        rx_ntn_z = np.where(
            self.cm.bldg_grid[filtered_indices[:, 0], filtered_indices[:, 1]],
            self.cm.zmax_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_roof,
            self.cm.zmin_grid[filtered_indices[:, 0], filtered_indices[:, 1]] + self.ntn_height_above_ground
        )
        self.rx_ntn_pos = np.column_stack((rx_ntn_x, rx_ntn_y, rx_ntn_z))
        
        
        # One fixed indoor/outdoor draw per rectangle-sector region per macro.
        rng = np.random.default_rng(np.random.randint(0, 2**32, dtype=np.uint32))
        self.tn_sampler = SectorTNSampler(
            self.cm.x, self.cm.y, self.cm.bldg_grid, self.cm.zmin_grid, self.cm.zmax_grid,
            self.tx_pos, outdoor_probability=tn_outdoor_probability,
            yaw_offset_rad=tn_sector_yaw_offset_rad, rng=rng,
            max_attempts=tn_max_attempts, outdoor_height=self.tn_height_above_ground,
            indoor_roof_offset=self.tn_height_above_roof,
        )
        self.tn_outdoor_probability = float(tn_outdoor_probability)
        self.tn_association_margin_db = float(tn_association_margin_db)
        self.tn_min_channel_norm = float(tn_min_channel_norm)
        self.tn_candidates_per_batch = int(tn_candidates_per_batch)
        self.tn_drop_output_dir = tn_drop_output_dir
        self._tn_drop_macro_index = getattr(self, "_tn_drop_macro_index", -1) + 1
        self.tn_drop_yaw_rad = float(tn_sector_yaw_offset_rad)
        self.tn_pos, self.tn_target_tx_index = self.tn_sampler.draw(np.arange(12))
        self.tn_is_outdoor = self.tn_sampler.is_outdoor.copy()
        self.tn_bs_index = self.tn_target_tx_index // 3
        self.tn_sector_index = self.tn_target_tx_index % 3

        # 5) Compute random satellite direction & project it to bounding box
        # azimuth = np.random.uniform(0, 360)
        # elevation = np.random.uniform(25, 90)
        x_proj, y_proj, z_proj = satellite_projection(
            azimuth,
            elevation,
            self.sat_distance,
            self.L_NS,
            self.W_WE
        )
        self.ntn_look_pos = np.array([x_proj, y_proj, z_proj])

        if plot_grid or plot_bs or plot_tn or plot_ntn:
            self._plot_grid(show_bs=plot_bs, show_tn=plot_tn, show_ntn=plot_ntn)
        
        
        
        
        

    def compute_paths(self, nsect, fc, tx_rows = 8, tx_cols = 8, tn_rx_rows = 1, tn_rx_cols = 1, max_depth=3,
                      bandwidth=100e6, tx_power_dbm=30,
                      sector_yaw_offset_rad=None,
                      sector_pitch_rad=None,
                      sector_roll_rad=None,
                      ntn_los_mode="natural", propagation_options=None):
        """
        1) Configure scene frequency and remove old TX/RX
        2) Add TX, add TN array and receivers => compute TN CIR
        3) Remove TN, switch RX array to single-element custom => compute NTN CIR
        sector_*_rad:
            If None, use current object defaults:
            self.tx_sector_yaw_offset_rad / self.tx_sector_pitch_rad / self.tx_sector_roll_rad.
        """
        self.tn_solver_options, self.ntn_solver_options = resolve_propagation_options(
            max_depth, ntn_los_mode, propagation_options)
        max_depth = self.tn_solver_options["max_depth"]
        if sector_yaw_offset_rad is None:
            sector_yaw_offset_rad = self.tx_sector_yaw_offset_rad
        if sector_pitch_rad is None:
            sector_pitch_rad = self.tx_sector_pitch_rad
        if sector_roll_rad is None:
            sector_roll_rad = self.tx_sector_roll_rad

        sector_yaw_offset_rad = float(sector_yaw_offset_rad)
        sector_pitch_rad = float(sector_pitch_rad)
        sector_roll_rad = float(sector_roll_rad)

        if nsect != 3 or not hasattr(self, "tn_sampler"):
            raise ValueError("Run sector-drop compute_positions first and use nsect=3.")
        yaw_error = np.angle(np.exp(1j * (sector_yaw_offset_rad - self.tn_drop_yaw_rad)))
        if not np.isclose(yaw_error, 0.0, atol=1e-10):
            raise ValueError("TN partition yaw must match the sector yaw used for channels.")

        # Keep the latest values for future calls
        self.tx_sector_yaw_offset_rad = sector_yaw_offset_rad
        self.tx_sector_pitch_rad = sector_pitch_rad
        self.tx_sector_roll_rad = sector_roll_rad

        self.fc = fc
        self.nsect = nsect
        self.scene.frequency = self.fc
        self.scene.bandwidth = bandwidth
        # Remove existing TX and RX
        for rx_name in list(self.scene.receivers):
            self.scene.remove(rx_name)
        for tx_name in list(self.scene.transmitters):
            self.scene.remove(tx_name)

        # A. Set up the TX array
        self.scene.tx_array = PlanarArray(
            num_rows = tx_rows,
            num_cols = tx_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            polarization="V",
            pattern="tr38901"
        )

        # B. Set up the multi-element array for the TN side
        self.scene.rx_array = PlanarArray(
            num_rows = tn_rx_rows,
            num_cols = tn_rx_cols,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            polarization="V",
            # pattern="tr38901"
            pattern="dipole"
        )

        # (1) Add Transmitters
        #    Use multiple sector approach for the single base station
        sector_yaw = np.mod(
            float(sector_yaw_offset_rad) + 2.0 * np.pi * np.arange(self.nsect) / self.nsect,
            2.0 * np.pi,
        )
        tx_name_list = []
        tx_bs_index = []
        tx_sector_index = []
        tx_orientation = []
        for i in range(self.nbs):
            for s in range(self.nsect):
                yaw = float(sector_yaw[s])
                name = f"tx-{i}-{s}"
                tx = sionna.rt.Transmitter(
                    name=name,
                    position=self.tx_pos[i],
                    power_dbm=tx_power_dbm,
                    orientation=[yaw, float(sector_pitch_rad), float(sector_roll_rad)]
                )
                self.scene.add(tx)
                tx_name_list.append(name)
                tx_bs_index.append(int(i))
                tx_sector_index.append(int(s))
                tx_orientation.append([yaw, float(sector_pitch_rad), float(sector_roll_rad)])
        self.tx_name_list = tx_name_list
        self.tx_bs_index = np.asarray(tx_bs_index, dtype=int)
        self.tx_sector_index = np.asarray(tx_sector_index, dtype=int)
        self.tx_orientation_rad = np.asarray(tx_orientation, dtype=float)

        p_solver = PathSolver()
        self._compute_sector_tn_paths(p_solver, max_depth)

        if self.ntn_rx > 0:
            for rx_name in list(self.scene.receivers):
                self.scene.remove(rx_name)

            if self.ntn_look_pos is None:
                raise ValueError("ntn_look_pos is empty; run compute_positions before compute_paths.")

            self.scene.rx_array = PlanarArray(
                num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="vsat_dish",
                polarization="V"
            )

            for i in range(self.rx_ntn_pos.shape[0]):
                rx = sionna.rt.Receiver(
                    name=f"ntn-{i}",
                    color=[1.0, 0.0, 0.0],
                    position=self.rx_ntn_pos[i]
                )
                self.scene.add(rx)
                # All NTN users share the projected satellite look-at point
                # for the current macro simulation.
                # rx.look_at(self.ntn_look_pos)
                rx.look_at(self.ntn_look_pos+self.rx_ntn_pos[i])

            
            self.paths_ntn = p_solver(scene=self.scene, **self.ntn_solver_options)

            # Compute paths for TN

            self.a_ntn, self.tau_ntn = self.paths_ntn.cir(normalize_delays=False, out_type="numpy")

    def compute_ntn_ul_paths(self, frequency_hz, *, max_depth=3):
        """Trace the reciprocal NTN link at UL frequency with fixed BS geometry.

        Run compute_paths at the DL reference frequency first. The same endpoints,
        orientations, element patterns and single-element NTN receiver are used.
        Sionna traces BS->NTN; reciprocity supplies the corresponding UL channel
        in the existing TX-array column-vector convention. No TNs are redropped.
        Only electrical spacing changes: p_UL/lambda_UL =
        (f_UL/f_DL) * p_DL/lambda_DL. DL scene state is restored even on failure.
        """
        frequency_hz = float(frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("UL frequency must be finite and positive.")
        if self.ntn_rx <= 0 or not hasattr(self, "a_ntn"):
            raise ValueError("Trace a nonempty NTN DL scene before UL sensing.")
        if frequency_hz == float(self.fc):
            return self.paths_ntn, self.a_ntn, self.tau_ntn
        if set(self.scene.receivers) != {f"ntn-{i}" for i in range(len(self.rx_ntn_pos))}:
            raise ValueError("UL sensing requires the retained NTN receivers from compute_paths.")
        dl_frequency = float(np.asarray(self.scene.frequency).item())
        array = self.scene.tx_array
        dl_positions = array.normalized_positions
        try:
            self.scene.frequency = frequency_hz
            array.normalized_positions = dl_positions * (frequency_hz / dl_frequency)
            # Match the DL depth, LOS policy, propagation mechanisms and seed.
            options = getattr(self, "ntn_solver_options", dict(
                max_depth=max_depth, los=True, specular_reflection=True,
                diffuse_reflection=False, refraction=True, synthetic_array=True))
            paths = PathSolver()(scene=self.scene, **options)
            a, tau = paths.cir(normalize_delays=False, out_type="numpy")
            return paths, a, tau
        finally:
            array.normalized_positions = dl_positions
            self.scene.frequency = dl_frequency

    def _trace_tn_candidates(self, p_solver, max_depth):
        for name in list(self.scene.receivers):
            self.scene.remove(name)
        self.tn_bs_index = self.tn_target_tx_index // 3
        self.tn_sector_index = self.tn_target_tx_index % 3
        for i, pos in enumerate(self.tn_pos):
            rx = sionna.rt.Receiver(name=f"tn-{i}", position=pos, color=[0.0, 1.0, 0.0])
            self.scene.add(rx)
            rx.look_at(self.tx_pos[self.tn_bs_index[i]])
        options = getattr(self, "tn_solver_options", dict(
            max_depth=max_depth, los=True, specular_reflection=True,
            diffuse_reflection=False, refraction=True, synthetic_array=True))
        paths = p_solver(scene=self.scene, **options)
        a, tau = paths.cir(normalize_delays=False, out_type="numpy")
        return paths, a, tau

    def _save_tn_drop_report(self, status, accepted, best_margin, **extra):
        report = dict(
            status=status, outdoor_probability=self.tn_outdoor_probability,
            association_margin_db=self.tn_association_margin_db,
            min_channel_norm=self.tn_min_channel_norm,
            rectangles_xy=self.tn_sampler.rectangles.tolist(),
            target_tx_index=list(range(12)),
            is_outdoor=self.tn_sampler.is_outdoor.tolist(),
            draw_counts=self.tn_sampler.draw_counts.tolist(),
            accepted=accepted.tolist(),
            best_margin_db=[float(v) if np.isfinite(v) else None for v in best_margin],
            **extra,
        )
        self.tn_drop_diagnostics = report
        if self.tn_drop_output_dir is not None:
            directory = Path(self.tn_drop_output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"drop_{self._tn_drop_macro_index:04d}.json"
            with path.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)

    def _compute_sector_tn_paths(self, p_solver, max_depth):
        accepted = np.zeros(12, dtype=bool)
        positions = np.zeros((12, 3), dtype=float)
        best_margin = np.full(12, -np.inf)
        while True:
            _, a, _ = self._trace_tn_candidates(p_solver, max_depth)
            h = collapse_cir_to_narrowband(a)
            valid, margin, _, _ = channel_dominance(
                h, self.tn_target_tx_index, self.tn_association_margin_db,
                self.tn_min_channel_norm,
            )
            for i, target in enumerate(self.tn_target_tx_index):
                best_margin[target] = max(best_margin[target], margin[i])
                if valid[i] and not accepted[target]:
                    positions[target] = self.tn_pos[i]
                    accepted[target] = True
            print(
                f"[Sector TN drop] accepted={accepted.sum()}/12 "
                f"candidates={self.tn_sampler.draw_counts.sum()} "
                f"outdoor={self.tn_sampler.is_outdoor.sum()}/12 "
                f"margin_required={self.tn_association_margin_db:g} dB",
                flush=True,
            )
            pending = np.flatnonzero(~accepted)
            if len(pending) == 0:
                break
            exhausted = self.tn_sampler.exhausted(pending)
            if exhausted:
                self._save_tn_drop_report("failed", accepted, best_margin)
                details = [
                    f"BS {t // 3}/sector {t % 3} "
                    f"({'outdoor' if self.tn_sampler.is_outdoor[t] else 'indoor'}, "
                    f"tried={self.tn_sampler.draw_counts[t]}, best_margin={best_margin[t]:g} dB)"
                    for t in exhausted
                ]
                raise RuntimeError(
                    "No feasible sector TN within the candidate budget: " + "; ".join(details)
                    + ". No UE type or target sector was changed. Indoor points are below "
                    "the roof and may have zero channels with max_depth=0. Check propagation, "
                    "margin, minimum channel norm, or increase tn_max_attempts."
                )
            self.tn_pos, self.tn_target_tx_index = self.tn_sampler.draw(
                pending, self.tn_candidates_per_batch,
            )

        # Re-trace only the twelve retained UEs, keeping paths/CIR/positions aligned.
        self.tn_pos = positions
        self.tn_target_tx_index = np.arange(12)
        self.paths_tn, self.a_tn, self.tau_tn = self._trace_tn_candidates(p_solver, max_depth)
        h = collapse_cir_to_narrowband(self.a_tn)
        valid, margin, serving, competitor = channel_dominance(
            h, self.tn_target_tx_index, self.tn_association_margin_db,
            self.tn_min_channel_norm,
        )
        if not np.all(valid):
            self._save_tn_drop_report("final_validation_failed", valid, best_margin)
            raise RuntimeError("Final TN channel re-trace failed the fixed-sector dominance check.")
        self.tn_is_outdoor = self.tn_sampler.is_outdoor.copy()
        self.tn_association_margin_actual_db = margin
        self._save_tn_drop_report(
            "success", accepted, best_margin, positions_xyz=self.tn_pos.tolist(),
            achieved_margin_db=[float(v) if np.isfinite(v) else None for v in margin],
            zero_competitor_power=(competitor == 0).tolist(),
            serving_channel_power=serving.tolist(), strongest_other_power=competitor.tolist(),
        )
