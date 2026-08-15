import cv2
import numpy as np

class Stitcher:
    """
    Stitcher class:
        feature_type (int): feature type (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB, 4: SIFT). The default is 3 (ORB).
        match_thres (float): confidence for feature matching step. The default value is 0.3 for ORB and 0.65 for other feature types.
        matcher_type (int): matcher type used for pairwise image matching (0: homography, 1: affine). The default is 0 (homography).
        conf_thres (float): threshold for two imgs are from the same panorama confidence. The default is 1.0.
        work_megapix (float): resolution (Mpx) for image registration step. The default is 0.6.
        seam_megapix (float): resolution (Mpx) for seam estimation step. The default is 0.1.
        seam_choice (int): seam estimation type (0: dp_color, 1: dp_colorgrad, 2: voronoi, 3: no). The default is 2 (voronoi).
        wave_correct_choice (int): wave effect correction type (0: horizontal, 1: none, 2: vertical). The default is 1 (none).
        expose_comp_choice (int): exposure compensation method type (0: gain_blocks, 1: gain, 2: channel, 3: channel_blocks, 4: no). The default is 0 (gain_block)
        expos_comp_nr_feeds (int): number of exposure compensation feed. The default is 1.
        expos_comp_nr_filtering (float): number of filtering iterations of the exposure compensation gains. The default is 2.0.
        expos_comp_block_size (int): block size in pixels used by the exposure compensator. The default is 32.
        blend_choice (int): blending method type (0: multiband, 1: feather, 2: no). The default is 0 (multiband).
        blend_strength (int): blending strength from [0,100] range. The default is 5.
        estimator (int): transformation estimator type (0: homography, 1: affine). The default is 0 (homography).
        ba (int): bundle adjustment type (0: ray, 1: reproj, 2: affine, 3: no). The default is 0 (ray).
        ba_refine_mask (str): set refinement mask for bundle adjustment. It looks like 'x_xxx' 
            where 'x' means refine respective parameter and '_' means don't refine,
            and has the following format:<fx><skew><ppx><aspect><ppy>.
            The default mask is 'xxxxx'.
        warp_surface_type (str): warp surface type ('spherical', 'plane', 'cylindrical'...). The default is 'spherical'
    """
    def __init__(self, feature_type = 3, match_thres = -1, matcher_type = 0, conf_thres = 1.0, 
                 work_megapix = 0.6, seam_megapix = 0.1, seam_choice = 2, 
                 wave_correct_choice = 1, 
                 expose_comp_choice = 0, expos_comp_nr_feeds = 1, expos_comp_nr_filtering = 2.0, expos_comp_block_size = 32,
                 blend_choice = 0, blend_strength = 5, 
                 estimator = 0, ba = 0,
                 ba_refine_mask ='xxxxxx', warp_surface_type = 'plane', try_use_gpu = True):
        """Initialize Stitcher object

        Args:
            feature_type (int): feature type (0: AKAZE, 1: BRISK, 2: KAZE, 3: ORB, 4: SIFT). The default is 3 (ORB).
            match_thres (float): confidence for feature matching step. The default value is 0.3 for ORB and 0.65 for other feature types.
            matcher_type (int): matcher type used for pairwise image matching (0: homography, 1: affine). The default is 0 (homography).
            conf_thres (float): threshold for two imgs are from the same panorama confidence. The default is 1.0.
            work_megapix (float): resolution (Mpx) for image registration step. The default is 0.6.
            seam_megapix (float): resolution (Mpx) for seam estimation step. The default is 0.1.
            seam_choice (int): seam estimation type (0: dp_color, 1: dp_colorgrad, 2: voronoi, 3: no). The default is 2 (voronoi).
            wave_correct_choice (int): wave effect correction type (0: horizontal, 1: none, 2: vertical). The default is 1 (none).
            expose_comp_choice (int): exposure compensation method type (0: gain_blocks, 1: gain, 2: channel, 3: channel_blocks, 4: no). The default is 0 (gain_block)
            expos_comp_nr_feeds (int): number of exposure compensation feed. The default is 1.
            expos_comp_nr_filtering (float): number of filtering iterations of the exposure compensation gains. The default is 2.0.
            expos_comp_block_size (int): block size in pixels used by the exposure compensator. The default is 32.
            blend_choice (int): blending method type (0: multiband, 1: feather, 2: no). The default is 0 (multiband).
            blend_strength (int): blending strength from [0,100] range. The default is 5.
            estimator (int): transformation estimator type (0: homography, 1: affine). The default is 0 (homography).
            ba (int): bundle adjustment type (0: ray, 1: reproj, 2: affine, 3: no). The default is 0 (ray).
            ba_refine_mask (str): set refinement mask for bundle adjustment. It looks like 'x_xxx' 
                where 'x' means refine respective parameter and '_' means don't refine,
                and has the following format:<fx><skew><ppx><aspect><ppy>.
                The default mask is 'xxxxx'.
            warp_surface_type (str): warp surface type ('spherical', 'plane', 'cylindrical'...). The default is 'spherical'
        """        
        
        # feature extractor
        if feature_type == 3:
            self.extractor = cv2.ORB.create()
        else:
            if feature_type == 0:
                self.extractor = cv2.AKAZE_create()
            elif feature_type == 1:
                self.extractor = cv2.BRISK_create()
            elif feature_type == 2:
                self.extractor = cv2.KAZE.create()
            elif feature_type == 4:
                self.extractor = cv2.SIFT_create()
            else:
                try:
                    self.extractor = cv2.xfeatures2d_SURF.create()
                except:
                    print("SURF not available")
                    self.extractor = cv2.ORB.create()
                    feature_type = 3
        
        # feature matching threshold
        if match_thres >= 0 and match_thres <= 1:
            self.matchThres = match_thres
        else:
            if feature_type == 3:
                self.matchThres = 0.3
            else:
                self.matchThres = 0.65
        
        # feature matcher
        if matcher_type == 0:
            self.matcher = cv2.detail_BestOf2NearestMatcher(try_use_gpu, self.matchThres)
        else:
            self.matcher = cv2.detail_AffineBestOf2NearestMatcher(False, try_use_gpu, self.matchThres)
        
        # threshold for two imgs are from the same panorama confidence
        self.confThres = conf_thres
        
        # resolution (Mpx) for image registration step
        self.workMegapix = work_megapix

        # resolution (Mpx) for seam estimation step
        self.seamMegapix = seam_megapix

        # seam finder according to seam type
        if seam_choice == 0:
            self.seamFinder = cv2.detail_DpSeamFinder('COLOR')
        elif seam_choice == 1:
            self.seamFinder = cv2.detail_DpSeamFinder('COLOR_GRAD')
        elif seam_choice == 2:
            self.seamFinder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
        else:
            self.seamFinder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
        
        # wave effect correction
        if wave_correct_choice == 0:
            self.waveCorrect = cv2.detail.WAVE_CORRECT_HORIZ
        elif wave_correct_choice == 1:
            self.waveCorrect = None
        else:
            self.waveCorrect = cv2.detail.WAVE_CORRECT_VERT

        # exposure compensator
        if expose_comp_choice == 0:
            self.exposeComp = cv2.detail.ExposureCompensator_GAIN_BLOCKS
        elif expose_comp_choice == 1:
            self.exposeComp = cv2.detail.ExposureCompensator_GAIN
        elif expose_comp_choice == 2:
            self.exposeComp = cv2.detail.ExposureCompensator_CHANNELS 
        elif expose_comp_choice == 3:
            self.exposeComp = cv2.detail.ExposureCompensator_CHANNELS_BLOCKS 
        else:
            self.exposeComp = cv2.detail.ExposureCompensator_NO 

        # number of exposure compensation feed
        self.exposeCompNrFeeds = expos_comp_nr_feeds
        
        # number of filtering iterations of the exposure compensation gains
        self.exposeCompNrFiltering = expos_comp_nr_filtering
        
        # block size in pixels used by the exposure compensator
        self.exposeCompBlockSize = expos_comp_block_size
        
        # transformation estimator
        if estimator == 0:
            self.estimator = cv2.detail_HomographyBasedEstimator() # homography
        else:
            self.estimator = cv2.detail_AffineBasedEstimator()
        
        # bundle adjustment
        if ba == 0:
            self.adjuster = cv2.detail_BundleAdjusterRay() # ray
        elif ba == 1:
            self.adjuster = cv2.detail_BundleAdjusterReproj()
        elif ba == 2:
            self.adjuster = cv2.detail_BundleAdjusterAffinePartial()
        else:
            self.adjuster = cv2.detail_NoBundleAdjuster()
        self.adjuster.setConfThresh(1)
        
        refine_mask = np.zeros((3, 3), np.uint8)
        if ba_refine_mask[0] == 'x':
            refine_mask[0, 0] = 1
        if ba_refine_mask[1] == 'x':
            refine_mask[0, 1] = 1
        if ba_refine_mask[2] == 'x':
            refine_mask[0, 2] = 1
        if ba_refine_mask[3] == 'x':
            refine_mask[1, 1] = 1
        if ba_refine_mask[4] == 'x':
            refine_mask[1, 2] = 1
        self.adjuster.setRefinementMask(refine_mask)
        
        # blending method
        if blend_choice == 0:
            self.blenderType = 'multiband'
        elif blend_choice == 1:
            self.blenderType = 'feather'
        else:
            self.blenderType = 'no'
        
        #blending strength
        self.blendStrength = blend_strength
        if self.blendStrength < 0:
            self.blendStrength = 0
        elif self.blendStrength > 100:
            self.blendStrength = 100
        
        # warp surface type
        self.warpSurfaceType = warp_surface_type
        
        #cache cameras
        self.cacheCameras = None
        self.cacheSeamKs = list()#None
        
        self.result = None
        
        self.cacheCorners = list() #None
                
        self.cacheDstSz = None
        
        self.cacheBlendWidth = None
        
        self.cacheWarpedMasks1 = list()# None
        
        self.imgNum = 0
        
        self.cacheWarpedMasks2 = list()
        
    
    def getCompensator(self):
        """Generate exposure compensator

        Returns:
            compensator
        """        
        if self.exposeComp == cv2.detail.ExposureCompensator_CHANNELS:
            compensator = cv2.detail_ChannelsCompensator(self.exposeCompNrFeeds)
        elif self.exposeComp == cv2.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv2.detail_BlocksChannelsCompensator(
                self.exposeCompBlockSize, self.exposeCompBlockSize, self.exposeCompNrFeeds
            ) 
        else:            
            compensator = cv2.detail.ExposureCompensator_createDefault(self.exposeComp)
        return compensator
    
    def getBlender(self):
        """Generate warping blender

        Returns:
            blender
        """        
        blender = None
        if self.blenderType == "no" or self.cacheBlendWidth < 1:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        elif self.blenderType == "multiband":
            blender = cv2.detail_MultiBandBlender()
            blender.setNumBands((np.log(self.cacheBlendWidth) / np.log(2.) - 1.).astype(np.int32))
        elif self.blenderType == "feather":
            blender = cv2.detail_FeatherBlender()
            blender.setSharpness(1. / self.cacheBlendWidth)
        blender.prepare(self.cacheDstSz)
        return blender
    
    def registrate(self, imgs):
        """Image registration

        Args:
            imgs: input image list to extract features
        """        
        
        if self.workMegapix < 0:
            workScale = 1
        else:
            workScale = min(1.0, np.sqrt(self.workMegapix * 1e6 / (imgs[0].shape[1] * imgs[0].shape[0])))

        if self.seamMegapix > 0:
            self.cacheSeamScale = min(1.0, np.sqrt(self.seamMegapix * 1e6 / (imgs[0].shape[1] * imgs[0].shape[0])))
        else:
            self.cacheSeamScale = 1.0
            
        seamWorkAspect = self.cacheSeamScale / workScale#min(1.0, np.sqrt(self.seamMegapix * 1e6 / (resolution[0] * resolution[1]))) / min(1.0, np.sqrt(self.workMegapix * 1e6 / (resolution[0] * resolution[1])))

        # Step 1.1: Find features
        features = list()
        
        #i = 0
        for img in imgs:
            img2 = cv2.resize(src=img, dsize=None, fx=workScale, fy=workScale, interpolation=cv2.INTER_LINEAR_EXACT)
            img_feat = cv2.detail.computeImageFeatures2(self.extractor, img2)
            features.append(img_feat)
            
        #Step 1.2: Match features 
        p = self.matcher.apply2(features)
        self.matcher.collectGarbage()
            
        # Step 1.3: Select images & match subsets to build panorama
        indices = cv2.detail.leaveBiggestComponent(features, p, self.confThres)
        if len(indices) < self.imgNum:
            print("Contribution error")
            return
        
        # Step 1.4: Estimate homography matrix
        b, self.cacheCameras = self.estimator.apply(features, p, None)
        if not b:
            self.cacheCameras = None
            print("Homography estimation error")
            return
        for cam in self.cacheCameras:
            cam.R = cam.R.astype(np.float32)
            
        # Step 1.5: Refine camera parameters globally
            
        # Step 1.5.1: Final panorama scale estimation (bundle adjustment)
        b, self.cacheCameras = self.adjuster.apply(features, p, self.cacheCameras)
        if not b:
            self.cacheCameras = None
            print("Bundle adjustment error")
            return
            
        # Step 1.5.2: Wave correction
        if self.waveCorrect is not None:
            rmats = list()
            for cam in self.cacheCameras:
                rmats.append(np.copy(cam.R))
            rmats = cv2.detail.waveCorrect(rmats, self.waveCorrect)
            for idx, cam in enumerate(self.cacheCameras):
                cam.R = rmats[idx]

        # Step 1.6: Cache warped params estimation
        focals = list()
        for cam in self.cacheCameras:
            focals.append(cam.focal)
        focals.sort()
        if len(focals) % 2 == 1:
            warpedImageScale = focals[len(focals) // 2]
        else:
            warpedImageScale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
            
        composeWorkAspect = 1 / workScale

        self.cacheWarper1 = cv2.PyRotationWarper(self.warpSurfaceType, warpedImageScale * seamWorkAspect)  # warper could be nullptr?
        self.cacheWarper2 = cv2.PyRotationWarper(self.warpSurfaceType, warpedImageScale * composeWorkAspect)

        self.cacheWarperValue1 = warpedImageScale * seamWorkAspect
        self.cacheWarperValue2 = warpedImageScale * composeWorkAspect
        
        sizes = list()
        
        for i in range(self.imgNum):
            K = self.cacheCameras[i].K().astype(np.float32)
            K[0, 0] *= seamWorkAspect
            K[0, 2] *= seamWorkAspect
            K[1, 1] *= seamWorkAspect
            K[1, 2] *= seamWorkAspect
            self.cacheSeamKs.append(K)

            self.cacheCameras[i].focal *= composeWorkAspect
            self.cacheCameras[i].ppx *= composeWorkAspect
            self.cacheCameras[i].ppy *= composeWorkAspect
            
            K = self.cacheCameras[i].K().astype(np.float32)
            sz = (int(round(imgs[i].shape[1])), int(round(imgs[i].shape[0])))
            roi = self.cacheWarper2.warpRoi(sz, K, self.cacheCameras[i].R)
            self.cacheCorners[i] = roi[0:2]
            sizes.append(roi[2:4])
            
            mask = 255 * np.ones((imgs[i].shape[0], imgs[i].shape[1]), np.uint8)
            _, mask_warped = self.cacheWarper2.warp(mask, K, self.cacheCameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
            self.cacheWarpedMasks2[i] = mask_warped

        self.cacheDstSz = cv2.detail.resultRoi(corners=self.cacheCorners, sizes=sizes)
        self.cacheBlendWidth = np.sqrt(self.cacheDstSz[2] * self.cacheDstSz[3]) * self.blendStrength / 100        

    def composite(self, img0s):
        """Image compositing, assuming that camera parameters are determined

        Args:
            img0s: input image list

        Returns:
            result: stitched image
            dst: stitched image for display
        """        
        
        corners = list()
        masks_warped = list()
        images_warped = list()
        images_warped_f = list()

        for i in range(self.imgNum):
            img = cv2.resize(src=img0s[i], dsize=None, fx=self.cacheSeamScale, fy=self.cacheSeamScale, interpolation=cv2.INTER_LINEAR_EXACT)
            corner, image_wp = self.cacheWarper1.warp(img, self.cacheSeamKs[i], self.cacheCameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            corners.append(corner)
            images_warped.append(image_wp)
            images_warped_f.append(image_wp.astype(np.float32))
            if self.cacheWarpedMasks1[i] is None:
                um = cv2.UMat(255 * np.ones((img.shape[0], img.shape[1]), np.uint8))
                _, mask_wp = self.cacheWarper1.warp(um, self.cacheSeamKs[i], self.cacheCameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
                self.cacheWarpedMasks1[i] = mask_wp.get()

        # # Step 2.0: Preprocess images
        # imgs = list()
        # for img0 in img0s:
        #     img = cv2.resize(src=img0, dsize=None, fx=self.cacheSeamScale, fy=self.cacheSeamScale, interpolation=cv2.INTER_LINEAR_EXACT)
        #     imgs.append(img)
        
        # # Step 2.1: Warp images
        # corners = list()
        # masks_warped = list()
        # images_warped = list()
        # images_warped_f = list()
        
        # for i in range(self.imgNum):
        #     corner, image_wp = self.cacheWarper1.warp(imgs[i], self.cacheSeamKs[i], self.cacheCameras[i].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        #     corners.append(corner)
        #     images_warped.append(image_wp)
        #     images_warped_f.append(image_wp.astype(np.float32))
        #     if self.cacheWarpedMasks1[i] is None:
        #         um = cv2.UMat(255 * np.ones((imgs[i].shape[0], imgs[i].shape[1]), np.uint8))
        #         _, mask_wp = self.cacheWarper1.warp(um, self.cacheSeamKs[i], self.cacheCameras[i].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        #         self.cacheWarpedMasks1[i] = mask_wp.get()
        
        # Step 2.2: Estimate exposure errors 
        compensator = self.getCompensator()
        compensator.feed(corners=corners, images=images_warped, masks=self.cacheWarpedMasks1)
        
        # Step 2.3: Find seam masks
        masks_warped = self.seamFinder.find(images_warped_f, corners, self.cacheWarpedMasks1)

        # Step 2.4 - 2.5: Compensate exposure error - Blend 
        blender = self.getBlender()
        for idx in range(self.imgNum):
            K = self.cacheCameras[idx].K().astype(np.float32)
            corner, image_warped = self.cacheWarper2.warp(img0s[idx], K, self.cacheCameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
            compensator.apply(idx, self.cacheCorners[idx], image_warped, self.cacheWarpedMasks2[idx])
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv2.dilate(masks_warped[idx], None)
            # Resize seam mask
            seam_mask = cv2.resize(dilated_mask, (self.cacheWarpedMasks2[idx].shape[1], self.cacheWarpedMasks2[idx].shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
            mask_warped = cv2.bitwise_and(seam_mask, self.cacheWarpedMasks2[idx])
            blender.feed(cv2.UMat(image_warped_s), mask_warped, self.cacheCorners[idx])
         
        result, _ = blender.blend(None, None)
        result = cv2.convertScaleAbs(result)
        #result = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return result
    
    def saveCache(self):
        """Save cache stitching params
        """        
        for i in range(self.imgNum):
            np.save("cacheCameras_focal_{}.npy".format(i), self.cacheCameras[i].focal)
            np.save("cacheCameras_aspect_{}.npy".format(i), self.cacheCameras[i].aspect)
            np.save("cacheCameras_ppx_{}.npy".format(i), self.cacheCameras[i].ppx)
            np.save("cacheCameras_ppy_{}.npy".format(i), self.cacheCameras[i].ppy)
            np.save("cacheCameras_R_{}.npy".format(i), self.cacheCameras[i].R)
            np.save("cacheCameras_t_{}.npy".format(i), self.cacheCameras[i].t)
            np.save("cacheCorners_{}.npy".format(i), self.cacheCorners[i])
            np.save("cacheSeamKs_{}.npy".format(i), self.cacheSeamKs[i])
        
        np.save("cacheBlendWidth.npy", self.cacheBlendWidth)
        np.save("cacheDstSz.npy", self.cacheDstSz)
        np.save("cacheWarpedMasks1.npy", self.cacheWarpedMasks1, allow_pickle=True)
        np.save("cacheWarpedMasks2.npy", self.cacheWarpedMasks2, allow_pickle=True)
        np.save("cacheWarper1.npy", self.cacheWarperValue1)
        np.save("cacheWarper2.npy", self.cacheWarperValue2)
        np.save("cacheSeamScale.npy", self.cacheSeamScale)
        
    def loadCache(self):
        """Load cache stitching params
        """        
        self.cacheCameras = [None] * self.imgNum
        self.cacheCorners = [None] * self.imgNum
        self.cacheSeamKs = [None] * self.imgNum
        
        for i in range(self.imgNum):
            self.cacheCameras[i] = cv2.detail.CameraParams()
            self.cacheCameras[i].focal = float((np.load("cacheCameras_focal_{}.npy".format(i))))
            self.cacheCameras[i].aspect = float((np.load("cacheCameras_aspect_{}.npy".format(i))))
            self.cacheCameras[i].ppx = float(np.load("cacheCameras_ppx_{}.npy".format(i)))
            self.cacheCameras[i].ppy = float(np.load("cacheCameras_ppy_{}.npy".format(i)))
            self.cacheCameras[i].R = np.load("cacheCameras_R_{}.npy".format(i))
            self.cacheCameras[i].t = np.load("cacheCameras_t_{}.npy".format(i))
            self.cacheCorners[i] = tuple(np.load("cacheCorners_{}.npy".format(i)))
            self.cacheSeamKs[i] = np.load("cacheSeamKs_{}.npy".format(i))
            
        self.cacheBlendWidth = float(np.load("cacheBlendWidth.npy"))
        self.cacheDstSz = tuple(np.load("cacheDstSz.npy"))
        self.cacheWarpedMasks1 = np.load("cacheWarpedMasks1.npy", allow_pickle=True)
        self.cacheWarpedMasks2 = np.load("cacheWarpedMasks2.npy", allow_pickle=True)
        self.cacheWarper1 = cv2.PyRotationWarper(self.warpSurfaceType, float(np.load("cacheWarper1.npy")))  # warper could be nullptr?
        self.cacheWarper2 = cv2.PyRotationWarper(self.warpSurfaceType, float(np.load("cacheWarper2.npy")))
        self.cacheSeamScale = float(np.load("cacheSeamScale.npy"))
    
    def stitch(self, imgs):
        """Stitch images

        Args:
            imgs: input images (in order are left image, rear image, and right image)

        Returns: tuple (ret, frame)
            ret: stitching is success (True) or not (False)
            frame: stitched image
        """        
        
        if self.imgNum == 0:
            self.imgNum = len(imgs)
            self.cacheWarpedMasks1 = [None] * self.imgNum
            self.cacheWarpedMasks2 = [None] * self.imgNum
            self.cacheCorners = [None] * self.imgNum

        # Phase 1: Image Registration
        if self.cacheCameras is None:# or self.imgNum != len(imgs): 
            # if os.path.exists("cacheCameras_focal_0.npy"):
            #     self.loadCache()
            # else:
            self.registrate(imgs)
        
        # Phase 2: Compositing
        if self.cacheCameras is not None:# and self.imgNum != 0:
            # if not os.path.exists("cacheCameras_focal_0.npy"):
            #     self.saveCache()
            return True, self.composite(imgs)
            
        return False, None



        
            
        
