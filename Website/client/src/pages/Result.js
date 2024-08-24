import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import Chip from '@mui/material/Chip';
import ResponsiveAppBar from '../components/Navbar';

function Result() {
  const navigate = useNavigate();
  const location = useLocation();

  const [imagePreview, setImagePreview] = useState(false);
  const [prediction, setPrediction] = useState(false);
    
  useEffect(() => {
    if(location.state === null) navigate('/');
    
    const state = location.state;
    if(state) {
        setImagePreview(state.imagePreview);
        setPrediction(state.prediction);
    }

  }, []);

  const handleScanAgain = () => {
    navigate('/get-started');
  }
  
  return (
    <>
        <ResponsiveAppBar />
        <div style={{ backgroundColor: '#52B58812', paddingTop: '.5rem', paddingBottom: '16rem' }}>
            <Container>
                <div style = {{ display: 'grid', justifyContent: 'center', width: '100%', }}>
                    <div style={{ display: 'grid', border: '1px solid #c0c0c0', borderRadius: '5px', justifySelf: 'center', width: 'max-content' }}>
                        { imagePreview && <img src={imagePreview} style={{ padding: '1rem' }} /> }
                    </div>
                      <div style={{ justifySelf: 'center', marginTop: '1rem' }}>
                        {
                            prediction === 0 ?
                            <Chip style={{ fontFamily: 'Montserrat', fontSize: 14 }} label="Image Not Tampered" color="success" />
                            :
                            <Chip style={{ fontFamily: 'Montserrat', fontSize: 14 }} label="Image Tampered" color="error" />
                        }
                      </div>
                      <div style={{ justifySelf: 'center', marginTop: '1rem' }}>
                        <Button onClick={handleScanAgain} style={{ fontFamily: 'Montserrat', backgroundColor: '#14985C', color: 'white', textTransform: 'none' }}>
                            Scan Again
                        </Button>
                      </div>
                </div>
            </Container>
        </div>
    </>
  )
}

export default Result;