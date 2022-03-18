# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:36:43 2022

@author: user
"""
# Download the 56 zip files in Images_png in batches
import random
from urllib import request, error as urlerror
import hashlib
import os
import shutil
from tqdm import tqdm
from zipfile import ZipFile

from util.progressIterator import iterate as prog_iter

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip',
    'https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip',
    'https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip',
    'https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip',
    'https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip',
    'https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip',
    'https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip',
    'https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip',
    'https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip',
    'https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip',

    'https://nihcc.box.com/shared/static/2zsqpzru46wsp0f99eaag5yiad42iezz.zip',
    'https://nihcc.box.com/shared/static/8v8kfhgyngceiu6cr4sq1o8yftu8162m.zip',
    'https://nihcc.box.com/shared/static/jl8ic5cq84e1ijy6z8h52mhnzfqj36q6.zip',
    'https://nihcc.box.com/shared/static/un990ghdh14hp0k7zm8m4qkqrbc0qfu5.zip',
    'https://nihcc.box.com/shared/static/kxvbvri827o1ssl7l4ji1fngfe0pbt4p.zip',
    'https://nihcc.box.com/shared/static/h1jhw1bee3c08pgk537j02q6ue2brxmb.zip',
    'https://nihcc.box.com/shared/static/78hamrdfzjzevrxqfr95h1jqzdqndi19.zip',
    'https://nihcc.box.com/shared/static/kca6qlkgejyxtsgjgvyoku3z745wbgkc.zip',
    'https://nihcc.box.com/shared/static/e8yrtq31g0d8yhjrl6kjplffbsxoc5aw.zip',
    'https://nihcc.box.com/shared/static/vomu8feie1qembrsfy2yaq36cimvymj8.zip',

    'https://nihcc.box.com/shared/static/ecwyyx47p2jd621wt5c5tc92dselz9nx.zip',
    'https://nihcc.box.com/shared/static/fbnafa8rj00y0b5tq05wld0vbgvxnbpe.zip',
    'https://nihcc.box.com/shared/static/50v75duviqrhaj1h7a1v3gm6iv9d58en.zip',
    'https://nihcc.box.com/shared/static/oylbi4bmcnr2o65id2v9rfnqp16l3hp0.zip',
    'https://nihcc.box.com/shared/static/mw15sn09vriv3f1lrlnh3plz7pxt4hoo.zip',
    'https://nihcc.box.com/shared/static/zi68hd5o6dajgimnw5fiu7sh63kah5sd.zip',
    'https://nihcc.box.com/shared/static/3yiszde3vlklv4xoj1m7k0syqo3yy5ec.zip',
    'https://nihcc.box.com/shared/static/w2v86eshepbix9u3813m70d8zqe735xq.zip',
    'https://nihcc.box.com/shared/static/0cf5w11yvecfq34sd09qol5atzk1a4ql.zip',
    'https://nihcc.box.com/shared/static/275en88yybbvzf7hhsbl6d7kghfxfshi.zip',

    'https://nihcc.box.com/shared/static/l52tpmmkgjlfa065ow8czhivhu5vx27n.zip',
    'https://nihcc.box.com/shared/static/p89awvi7nj0yov1l2o9hzi5l3q183lqe.zip',
    'https://nihcc.box.com/shared/static/or9m7tqbrayvtuppsm4epwsl9rog94o8.zip',
    'https://nihcc.box.com/shared/static/vuac680472w3r7i859b0ng7fcxf71wev.zip',
    'https://nihcc.box.com/shared/static/pllix2czjvoykgbd8syzq9gq5wkofps6.zip',
    'https://nihcc.box.com/shared/static/2dn2kipkkya5zuusll4jlyil3cqzboyk.zip',
    'https://nihcc.box.com/shared/static/peva7rpx9lww6zgpd0n8olpo3b2n05ft.zip',
    'https://nihcc.box.com/shared/static/2fda8akx3r3mhkts4v6mg3si7dipr7rg.zip',
    'https://nihcc.box.com/shared/static/ijd3kwljgpgynfwj0vhj5j5aurzjpwxp.zip',
    'https://nihcc.box.com/shared/static/nc6rwjixplkc5cx983mng9mwe99j8oa2.zip',

    'https://nihcc.box.com/shared/static/rhnfkwctdcb6y92gn7u98pept6qjfaud.zip',
    'https://nihcc.box.com/shared/static/7315e79xqm72osa4869oqkb2o0wayz6k.zip',
    'https://nihcc.box.com/shared/static/4nbwf4j9ejhm2ozv8mz3x9jcji6knhhk.zip',
    'https://nihcc.box.com/shared/static/1lhhx2uc7w14bt70de0bzcja199k62vn.zip',
    'https://nihcc.box.com/shared/static/guho09wmfnlpmg64npz78m4jg5oxqnbo.zip',
    'https://nihcc.box.com/shared/static/epu016ga5dh01s9ynlbioyjbi2dua02x.zip',
    'https://nihcc.box.com/shared/static/b4ebv95vpr55jqghf6bthg92vktocdkg.zip',
    'https://nihcc.box.com/shared/static/byl9pk2y727wpvk0pju4ls4oomz9du6t.zip',
    'https://nihcc.box.com/shared/static/kisfbpualo24dhby243nuyfr8bszkqg1.zip',
    'https://nihcc.box.com/shared/static/rs1s5ouk4l3icu1n6vyf63r2uhmnv6wz.zip',

    'https://nihcc.box.com/shared/static/7tvrneuqt4eq4q1d7lj0fnafn15hu9oj.zip',
    'https://nihcc.box.com/shared/static/gjo530t0dgeci3hizcfdvubr2n3mzmtu.zip',
    'https://nihcc.box.com/shared/static/7x4pvrdu0lhazj83sdee7nr0zj0s1t0v.zip',
    'https://nihcc.box.com/shared/static/z7s2zzdtxe696rlo16cqf5pxahpl8dup.zip',
    'https://nihcc.box.com/shared/static/shr998yp51gf2y5jj7jqxz2ht8lcbril.zip',
    'https://nihcc.box.com/shared/static/kqg4peb9j53ljhrxe3l3zrj4ac6xogif.zip'
]

md5_link = 'https://nihcc.box.com/shared/static/q0f8gy79q2spw96hs6o4jjjfsrg17t55.txt'


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
            Number of blocks just transferred [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download(start_index=0, end_index=len(links), to_check_md5=True, md5_web_fetch=True, max_tries=5):
    for idx, link, prog in prog_iter(list(enumerate(links))[start_index:end_index]):
        for _ in range(max_tries):
            try:
                fn = 'Images_png_%02d.zip' % (idx + 1)
                if os.path.exists(fn):
                    print(prog, '- file exists', fn, ', skipping')
                    break
                print(prog, '- downloading', fn, '...')
                with tqdm(unit='B', miniters=1, unit_scale=True, desc="Downloading") as t:
                    request.urlretrieve(link, fn, reporthook(t))  # download the zip file
            except urlerror.URLError:
                print(prog, f"- error downloading")
                continue
            else:
                break

    if to_check_md5:
        print("Download complete. checking the MD5 checksums")
        md5_dict = __parseMD5(webFetch=md5_web_fetch)
        for idx, link in list(enumerate(links))[start_index:end_index]:
            fn = 'Images_png_%02d.zip' % (idx + 1)
            if not __check_md5(fn, md5_dict[fn]):
                print(f"md5 check failed in file \"{fn}\"")
        print("Done md5 checks")
    else:
        print("Download complete. Please check the MD5 checksums")


def __check_md5(file_name, original_md5):
    # Open,close, read file and calculate MD5 on its contents 
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        md5_returned = hashlib.md5(data).hexdigest()

    # Finally compare original MD5 with freshly calculated
    return original_md5 == md5_returned


def __parseMD5(webFetch=True):
    lines = None
    if webFetch:
        lines = request.urlopen(md5_link)  # download the MD5 checksum file
    else:
        data = open("MD5_checksums.txt", 'r')
        lines = data.readlines()
        data.close()
    md5_pairs = []
    for line in lines:
        p = line.split()
        md5_pairs.append((p[1], p[0]))
    if webFetch:
        md5_pairs = map((lambda x: (x[0].decode(), x[1].decode())), md5_pairs)
        return dict(md5_pairs)
    else:
        return dict(md5_pairs)


def unzip_dataset(start_index=0, end_index=len(links), output_dir="dataset", delete_after_done=True):
    print("unzipping dataset")
    for idx, prog in prog_iter(list(range(len(links)))[start_index:end_index]):
        fn = 'Images_png_%02d.zip' % (idx + 1)
        if os.path.exists(fn):
            print(f"{prog} - unzipping {fn}...")
            __unzip_file(fn, output_dir)
            if delete_after_done:
                os.remove(fn)
    print("done unzipping")


def __unzip_file(file_to_extract, output_path):
    with ZipFile(file_to_extract, "r") as zip_ref:
        for file in tqdm(desc="Extracting", iterable=zip_ref.namelist(), unit="Files", total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=output_path)


def flatten_dataset(input_dir, output_dir):
    file_counter = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            in_file_path = os.path.join(root, f)
            out_file_name = os.path.split(root)[-1] + "_" + f
            out_file_path = os.path.join(output_dir, out_file_name)
            shutil.move(in_file_path, out_file_path)
        file_counter += len(files)
    print(f"done moving {file_counter} files to \"{output_dir}\"")
    shutil.rmtree(input_dir, ignore_errors=True)


def split_dataset(input_dir, train_dir, val_dir, train_ratio):
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
    f = []
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        f.extend(filenames)
        break
    random.shuffle(f)
    size1 = int(len(f) * train_ratio)
    l1 = f[:size1]
    l2 = f[size1:]

    for file_name in l1:
        in_file_path = os.path.join(input_dir, file_name)
        out_file_path = os.path.join(train_dir, file_name)
        shutil.move(in_file_path, out_file_path)
    print(f"done moving {len(l1)} files to \"{train_dir}\"")
    for file_name in l2:
        in_file_path = os.path.join(input_dir, file_name)
        out_file_path = os.path.join(val_dir, file_name)
        shutil.move(in_file_path, out_file_path)
    print(f"done moving {len(l2)} files to \"{val_dir}\"")
    shutil.rmtree(input_dir, ignore_errors=True)


def create_dataset(start_index=0, end_index=len(links), train_dir="dataset",val_dir="val",train_ratio=0.7):
    download(start_index=start_index, end_index=end_index)
    unzip_dataset(start_index=start_index, end_index=end_index, output_dir="tmp")
    if not os.path.exists("tmp2"):
        os.mkdir("tmp2")
    flatten_dataset("tmp", "tmp2")
    split_dataset(input_dir="tmp2",train_dir=train_dir,val_dir=val_dir,train_ratio=train_ratio)


if __name__ == '__main__':
    d = __parseMD5(False)
    print(d['Images_png_56.zip'])
