import React from 'react';
import { useNavigate } from 'react-router-dom';

import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';

import ResponsiveAppBar from '../components/Navbar';

function Home() {
    const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/get-started');
  };

  return (
    <>
        <ResponsiveAppBar />
        <div style={{ backgroundColor: '#52B58812', paddingTop: '3rem', paddingBottom: '17rem', }}>
            <Container style = {{ display: 'flex' }}>
				<div>
                    <div>
                        <h1 style={{ fontSize: '4rem', fontWeight: 600, lineHeight: "1.2em", color: '#212123', fontFamily: 'Montserrat', letterSpacing: '.2rem', marginBottom: '1rem' }}>Spot the Fake: Image Verification Tool</h1>
                    </div>
                    <div>
                        <p style={{ fontSize: '1rem', fontWeight: 300, fontFamily: 'Montserrat', lineHeight: '1.5em' }}>Discover the truth behind any image with TruthLens &#8211; upload your photo and find out if it&#8217;s been tampered with using our advanced ml technology.</p>
                    </div>
                    <Box sx={{ mt: 3 }}>
                        <Button onClick={handleGetStarted} style={{ backgroundColor: '#14985C', color: 'white', textTransform: 'none', fontFamily: 'Montserrat' }} variant='contained'>
                            Get Started
                        </Button>
                    </Box>
                </div>
                <div style={{ paddingLeft: '70px', alignSelf: 'flex-end' }}>
                    <img src="https://10web-site.ai/98/wp-content/uploads/sites/110/2023/05/dose-juice-sTPy-oeA3h0-unsplash_Yiy9SCjz.webp" width="80%" alt="" loading="lazy" />
                </div>
            </Container>
        </div>
    </>
  )
}

export default Home;