import json
import random

#val2017 데이터중 person 객체가 탐지된 데이터만 가져와서 database구축

with open('val2017_annotations/instances_val2017.json') as f:
    ann_json = json.load(f)

#person 객체의 category_id가 1이기 때문에 1인 image의 id를 person list에 추가
person = list(map(lambda ann: ann['image_id'] if ann['category_id']==1 else None, ann_json['annotations']))
person = list(filter(lambda c: c!=None, person))

#person만 있는 dataset을 구축하기 위한 json에 image_path 추가
dataset_json = {
    im['id']: {'image_path': im['coco_url']}
    for im in ann_json['images']
    if im['id'] in person
}

#이제 person list에 있는 image_id만 caption을 불러와서 dataset_json에 넣기

with open('val2017_annotations/captions_val2017.json') as f:
    caption = json.load(f)

for im in caption['annotations']:
    if im['image_id'] in person:
        im.pop('id',None)
        dataset_json[im['image_id']]['options']={im['caption']:True} #정답이기에 True

#앞에 각 {}의 num를 매겨줌.
keys = list(dataset_json.keys())
data_set = {i:dataset_json[key] for i , key in enumerate(keys,start=1)}

#하나의 image당 하나의 caption만 가져옴.(아래 랜덤 추출에서 동일한 image의 caption이 나오지 않도록 하기 위함)
caption = [list(data_set[image]['options'].keys())[0] for image in data_set]


for i in range(1,len(data_set)+1):
    
    false_list = caption[:i]+caption[i+1:] #실제 정답 이외의 caption만 list만듦.

    f_sen = random.sample(false_list,3) #false_list에서 임의로 3개 추출
    
    data_set[i]['options'][f_sen[0]]=False #3개 추출한 것들은 False
    data_set[i]['options'][f_sen[1]]=False
    data_set[i]['options'][f_sen[2]]=False
    
    #계속 같은 위치가 정답일 수 없으니 순서 shuffle후 다시 data_set에 입력
    op = list(data_set[i]['options'].keys())
    random.shuffle(op)
    ran_op = {key:data_set[i]['options'][key] for key in op}
    data_set[i]['options'] = ran_op
    
#dic => json
data_json = json.dumps(data_set)

#json 파일 작성
with open('quiz_dataset.json','w') as json_f:
    json_f.write(data_json)
