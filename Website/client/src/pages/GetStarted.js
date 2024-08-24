import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import { Backdrop, CircularProgress } from '@mui/material';

import { instance } from '../axios';

import { DropzoneArea } from "mui-file-dropzone";

import ResponsiveAppBar from '../components/Navbar';

function GetStarted() {
  const navigate = useNavigate();

  const [image, setImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (files) => {
    setImage(files[0]);
  };

  const handleSubmit = async () => {
    if(image === undefined) return;
    
    setIsLoading(true);
    const data = new FormData();
    data.append('file', image);

    instance.post('/classify', data)
    .then((res) => {
      let prediction = 0;
      if(res.data.message === "Image is tampered") prediction = 1;
      else if(res.data.message === "Image is not tampered") prediction = 0;
      setIsLoading(false);
      navigate('/result', { state: { prediction, imagePreview: URL.createObjectURL(image) } });
    });
  };

  return (
    <>
        <Backdrop
                sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                open={isLoading}
            >
            <p style={{ marginRight: '1rem', fontFamily: 'Montserrat', letterSpacing: '.2rem' }}>PROCESSING YOUR IMAGE</p><CircularProgress color="inherit" />
        </Backdrop>
        <ResponsiveAppBar />
        <div style={{ backgroundColor: '#52B58812', paddingTop: '.5rem', paddingBottom: '17rem' }}>
            <Container style = {{ display: 'flex', flexDirection: 'column' }}>
                <p style={{ fontSize: '1rem', fontWeight: 300, fontFamily: 'Montserrat', lineHeight: '1.5em' }}>Upload your image below to check if the image is tampered.</p>
                <DropzoneArea acceptedFiles={['image/*']} filesLimit={1} onChange={handleChange} />
                <Button onClick={handleSubmit} style={{ padding: '1rem', backgroundColor: '#14985C', color: 'white', textTransform: 'none', fontFamily: 'Montserrat' }} variant='contained'>
                    Submit
                </Button>
            </Container>
        </div>
    </>
  )
}

export default GetStarted;