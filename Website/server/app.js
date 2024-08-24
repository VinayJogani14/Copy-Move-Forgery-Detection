require("dotenv").config({ path: __dirname + "/.env" });
const express = require("express");
const app = express();
const cors = require("cors");
const logger = require("morgan");
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs/promises");

const { upload, redisClient } = require('./config');
const { deleteUserFiles } = require("./utils");

app.use(express.static(__dirname + '/public'));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(logger("dev"));
app.use(cors());

const PORT = process.env.PORT || 8000;

app.get("/api/test", (req, res) => res.send("Working fine!"));

app.post("/api/classify", upload.single('file'), async (req, res, next) => {
  const formData = new FormData();
  const fileName = req.file.filename;

  const file = await fs.readFile(req.file.path);

  formData.append('image', file, `${fileName}`);

  const ML_SERVER_BASE_URL = process.env.ML_SERVER_BASE_URL;
  const response = await axios.post(`${ML_SERVER_BASE_URL}/classify`, formData);

  const predictionValue = response.data.prediction;

  const imageStats = await redisClient.get("imageStats");
  const formattedImageStats = JSON.parse(imageStats);
  if (formattedImageStats === null) {
    const obj = {
      authentic: predictionValue === 0 ? 1 : 0,
      tampered: predictionValue === 1 ? 1 : 0,
    };
    await redisClient.set("imageStats", JSON.stringify(obj));
  }  
  else {
    const updatedImageStats = {
      authentic: predictionValue === 0 ? formattedImageStats.authentic + 1 : formattedImageStats.authentic,
      tampered: predictionValue === 1 ? formattedImageStats.tampered + 1 : formattedImageStats.tampered,
    };
    await redisClient.set("imageStats", JSON.stringify(updatedImageStats));
  }

  const predictionMessage = predictionValue === 0 ? 'Image is not tampered' : 'Image is tampered';

  return res.status(200).json({ message: predictionMessage });
});

app.get('/api/image-stats', async (req, res, next) => {
  const imageStats = await redisClient.get("imageStats");
  const formattedImageStats = JSON.parse(imageStats);
  if (formattedImageStats === null) {
    const obj = {
      authentic: 0,
      tampered: 0,
    };
    await redisClient.set("imageStats", JSON.stringify(obj));
  }
  return res.status(200).json({ message: formattedImageStats });
});


// delete files after every hour
setInterval(() => {
  deleteUserFiles();
}, 1 * 60 * 60 * 1000);

app.listen(PORT, () => {
  console.log(`Server running on PORT ${PORT}`);
  redisClient.connect().then(()=> {
    console.log('redis is connected')
  })

});
