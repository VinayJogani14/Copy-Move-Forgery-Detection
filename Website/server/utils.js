const fs = require('fs');
const path = require('path');

const deleteUserFiles = () => {
  const folderPath = 'server/user_images';

  fs.readdir(folderPath, (err, files) => {
    if (err) {
      console.error('Error reading folder:', err);
      return;
    }
    console.log(files);
    if (files.length > 0) {
      files.forEach((file) => {
        const filePath = path.join(folderPath, file);
  
        fs.unlink(filePath, (err) => {
          if (err) {
            console.error('Error deleting file:', err);
          }
        });
      });
      console.log('All user files deleted!');
    }
  });
};

module.exports = {
    deleteUserFiles,
}