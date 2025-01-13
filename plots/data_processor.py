import pandas as pd
import glob
import re
import os

class ExperimentDataProcessor:
    def __init__(self, base_path):
        """
        Args:
            base_path (str): The path to the folder that contains experiment subfolders.
                             Example: "logs/Ant-v5/exp-16/rkl/"
        """
        self.base_path = base_path
        # Use processed_data as the only place (also acts as cache).
        self.processed_dir = 'processed_data'
        os.makedirs(self.processed_dir, exist_ok=True)

    @staticmethod
    def _extract_q(folder_name):
        match = re.search(r'q(\d+)_', folder_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_clip(folder_name):
        match = re.search(r'qstd(\d+\.\d+)', folder_name)
        return float(match.group(1)) if match else None

    @staticmethod
    def _extract_seed(folder_name):
        match = re.search(r'seed(\d+)', folder_name)
        return int(match.group(1)) if match else None

    def _get_processed_path(self):
        """
        Generate a file path under `processed_data/` with the format:
            {env_name}_{exp_name}_{method}_data.csv

        Example:
            If base_path = "logs/Ant-v5/exp-16/rkl/",
            this might generate "processed_data/Ant-v5_exp-16_rkl_data.csv"
        """
        parts = self.base_path.strip('/').split('/')
        # e.g. parts = ['logs', 'Ant-v5', 'exp-16', 'rkl']
        env_name = parts[1]
        exp_name = parts[2]
        method = parts[3]
        return os.path.join(self.processed_dir, f'{env_name}_{exp_name}_{method}_data.csv')

    def load_data(self, use_cache=True):
        """
        Load data from all matching folders into a single DataFrame.
        If use_cache=True and the processed file exists, read from it.
        Otherwise, process the subfolders, then save the result to the single
        processed_data file.

        Returns:
            pd.DataFrame: Combined experiment data.
        """
        processed_path = self._get_processed_path()

        if use_cache and os.path.exists(processed_path):
            print(f"Loading cached data from {processed_path}")
            return pd.read_csv(processed_path)

        data_rows = []
        # Glob example: "logs/Ant-v5/exp-16/rkl/2024_*_seed*"
        folders = glob.glob(os.path.join(self.base_path, '*_seed*'))

        # Extract environment name from base_path (for consistent naming inside the DF)
        env_name = self.base_path.split('/')[1]

        for folder in folders:
            q = self._extract_q(folder)
            clip = self._extract_clip(folder)
            seed = self._extract_seed(folder)

            # print(f"Processing folder: {folder}, q={q}, clip={clip}, seed={seed}")


            # try:
            df = pd.read_csv(f'{folder}/progress.csv')
            # Add experiment metadata
            df['q'] = q
            df['clip'] = clip
            df['folder'] = folder

            if 'Env Steps' in df.columns:
                df['episode'] = df['Env Steps']
            elif 'Running Env Steps' in df.columns:
                df['episode'] = df['Running Env Steps']
            elif 'episode' in df.columns:
                pass
            elif 'Itration' in df.columns:
                df['episode'] = df['Itration'] * 5000
            elif 'Iteration' in df.columns:
                df['episode'] = df['Iteration'] * 5000

            df['environment'] = env_name
            df['seed'] = seed
            data_rows.append(df)
            # except Exception as e:
            #     print(f"Error reading file in folder: {folder}: {e}")

        if not data_rows:
            raise ValueError(f"No data found in {self.base_path}")

        combined_df = pd.concat(data_rows, ignore_index=True)

        # Save to processed_data (also acts as cache)
        combined_df.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")

        return combined_df

    @classmethod
    def process_experiments(cls, root_path):
        """
        Discover and process all experiment folders beneath `root_path`.
        Each experiment is saved into `processed_data` folder (the only folder).
        
        Args:
            root_path (str): Path containing multiple environment folders,
                             each containing multiple experiments, e.g. "logs/"
        """
        # Find all relevant method folders (e.g., rkl, cisl) under each exp-*
        exp_folders = []
        for env_folder in glob.glob(os.path.join(root_path, "*/")):
            for exp_folder in glob.glob(os.path.join(env_folder, "exp-*/")):
                for method_folder in glob.glob(os.path.join(exp_folder, "*/")):
                    if os.path.isdir(method_folder):
                        exp_folders.append(method_folder)

        print(f"Found {len(exp_folders)} experiment folders to process")

        for exp_folder in exp_folders:
            try:
                processor = cls(exp_folder)
                print(f"Processing {exp_folder}")
                # Load (and thus process) data. Force reprocessing with use_cache=False
                processor.load_data(use_cache=False)
                # Since we only store in processed_data, there's no second saving step.
            except Exception as e:
                print(f"Error processing {exp_folder}: {e}")


if __name__ == "__main__":
    root_path = "logs"  # e.g. "logs/Ant-v5/" or just "logs/" if multiple envs
    ExperimentDataProcessor.process_experiments(root_path)
