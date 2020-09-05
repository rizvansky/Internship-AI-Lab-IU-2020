#	this transform performs rotation of an image by an angle in range [-max_angle; max_angle]
#	and takes biggest rectangular area of image without borders
class RotationWithoutBorders(object):
	def __init__(self, max_angle):
        self.max_angle = max_angle
    
    
    def __call__(self, image):
        angle = random.uniform(-self.max_angle, self.max_angle)
        image = transform.functional.rotate(image, angle=angle)
        image = np.array(image)
        
        image_height, image_width = image.shape[0:2]
        
        tup = self.largest_rotated_rect(image_width, image_height, math.radians(angle))
        w = tup[0]
        h = tup[1]
        
        image = self.crop_around_center(image, *self.largest_rotated_rect(image_width, image_height, math.radians(angle)))
    
        return transform.ToPILImage()(np.uint8(image))
	
	
    def largest_rotated_rect(self, w, h, angle):
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )
    
    
    def crop_around_center(self, image, width, height):
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return np.array(image[y1:y2, x1:x2], dtype='float32')
    
   
#	this transform performs the translation of an image by scaling factor 'max_shift' and 
#	crops the image by removing black borders of image
class RandomShift(object):
    def __init__(self, max_shift):
        self.max_shift = max_shift
        
    def __call__(self, image):
        width = np.array(image).shape[0]
        height = np.array(image).shape[1]
        shift = random.uniform(-self.max_shift, self.max_shift)
        pixels_to_shift = int(shift * width)
        
        image = transform.functional.affine(
            img=image, angle=0, translate=(pixels_to_shift, pixels_to_shift), scale=1.0, shear=0.0
        )

        if pixels_to_shift < 0:
            image = transform.functional.crop(
                image, 0, 0, height + pixels_to_shift, width + pixels_to_shift
            )
        elif pixels_to_shift > 0:
            image = transform.functional.crop(
                image, pixels_to_shift, pixels_to_shift, height - pixels_to_shift, width - pixels_to_shift
            )

        return image
		

#	this function calculates mean and std of dataset		
def calculate_normalization(data_path):
    transforms = transform.Compose([
        transform.Resize((224, 224)),
        transform.Grayscale(),
        transform.ToTensor()
    ])
    
    dataset = torchvision.datasets.ImageFolder(data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(len(dataloader) * 4)
    mean = 0.
    std = 0.
    
    nb_samples = 0.

    for data, targets in dataloader:
        data = data.to('cuda')
        targets = targets.to('cuda')
        batch_samples = data.size(0)
        print(nb_samples)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    
    return mean, std