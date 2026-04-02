from PIL import Image
import os

def create_gif(image_folder, output_path, duration=500):
    # 獲取資料夾內所有圖片（過濾出 .png, .jpg, .jpeg）
    images = []
    files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for filename in files:
        file_path = os.path.join(image_folder, filename)
        images.append(Image.open(file_path))

    if not images:
        print("資料夾中沒有圖片！")
        return

    # 儲存為 GIF
    # duration: 每一幀的顯示時間（毫秒）
    # loop: 0 代表無限循環，1 代表播放一次
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"GIF 已成功儲存至: {output_path}")

# 使用範例
create_gif('to_make_gif', 'result.gif', duration=200)