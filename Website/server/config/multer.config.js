const multer = require('multer');
const fs = require('fs');

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const userImagesPath = 'server/user_images';
    
    if(!fs.existsSync(userImagesPath))
      fs.mkdirSync(userImagesPath);
    
    cb(null, userImagesPath)
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' +file.originalname)
  }
})

const upload = multer({ storage: storage });

module.exports = upload;
