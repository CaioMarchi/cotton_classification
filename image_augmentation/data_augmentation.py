class ImageDataAugmentation:
    def __init__(self, classes, path, num_batches):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        self.classes = classes
        self.data_augmentation = ImageDataGenerator(
            rotation_range= 20,
            width_shift_range= 0.2,
            height_shift_range= 0.2,
            shear_range= 0.2,
            zoom_range= 0.2,
            horizontal_flip= True,
            vertical_flip=True)
        self.path = path
        self.num_batches = num_batches

    def image_augmentation(self):
        import os
        from PIL import Image
        import numpy as np
        import tensorflow
        
        for class_name in self.classes:
            class_path = os.path.join(self.path, class_name)
            filenames = os.listdir(class_path)
    
            #percorrer as imagens na pasta de classe
            for filename in filenames:
                if filename == '.DS_Store':
                    continue
                else:
                    image_path = os.path.join(class_path, filename)
                    image = Image.open(image_path)
                    #image = cv2.imread(image_path)
                    
                    image = np.expand_dims(image, axis=0)
                #aplicar o data augmentation
                
                augmented_image = self.data_augmentation.flow(
                    x= image,
                    save_to_dir= os.path.join(self.path, class_name),
                    batch_size= self.num_batches,
                    save_prefix= "aug_",
                    save_format = 'jpg'
                )
                
                # aplicar o data augmentation nas imagens
                for i in range(self.num_batches):
                    #obter o próximo lote de imagens aumentadas
                    augmented_images = augmented_image.next()