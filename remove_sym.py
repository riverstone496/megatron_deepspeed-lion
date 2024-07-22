import os
import shutil

def resolve_symlinks(directory):
    for root, dirs, files in os.walk(directory):
        for name in dirs + files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                print('target',path)
                target = os.readlink(path)
                print('target',target)
                if os.path.exists(target):
                    os.unlink(path)
                    if os.path.isdir(target):
                        shutil.copytree(target, path)
                    else:
                        shutil.copy2(target, path)
                    print(f"シンボリックリンクを解消: {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("使用方法: python script.py <ディレクトリパス>")
        sys.exit(1)
    
    resolve_symlinks(sys.argv[1])