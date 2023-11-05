from pycocotools.coco import COCO
from mmdet.datasets import CocoDataset

from .builder import DATASETS


class _COCO(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cats = self.dataset['categories']
        ids = sorted([cat['id'] for cat in cats])
        return ids


@DATASETS.register_module()
class Objects365(CocoDataset):
    CLASSES = \
        ('Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp',
         'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
         'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
         'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower',
         'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots',
         'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt',
         'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker',
         'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool',
         'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum',
         'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle',
         'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned',
         'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel',
         'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed',
         'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple',
         'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck',
         'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock',
         'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger',
         'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine',
         'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle',
         'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane',
         'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage',
         'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone',
         'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane',
         'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
         'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza',
         'Elephant', 'Skateboard', 'Surfboard', 'Gun',
         'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot',
         'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper',
         'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks',
         'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board',
         'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder',
         'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball',
         'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle',
         'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck',
         'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club',
         'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear',
         'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong',
         'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask',
         'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide',
         'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee',
         'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon',
         'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon',
         'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog',
         'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer',
         'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple',
         'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle',
         'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone',
         'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion',
         'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom',
         'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
         'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese',
         'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue',
         'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap',
         'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut',
         'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
         'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate',
         'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker',
         'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal',
         'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin',
         'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill',
         'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi',
         'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case',
         'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop',
         'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle',
         'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster',
         'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling',
         'Table Tennis ')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = _COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids()

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos
