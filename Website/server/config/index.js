const upload = require('./multer.config');
const redisClient = require('./redis.config');

module.exports = {
    upload,
    redisClient,
}