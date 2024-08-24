import React, { useEffect, useState } from 'react';

import Container from '@mui/material/Container';
import ResponsiveAppBar from '../components/Navbar';

import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';
import { Backdrop, Box, Card, CircularProgress, Paper } from '@mui/material';
import { instance } from '../axios';

ChartJS.register(ArcElement, Tooltip, Legend);

function Analytics() {
  const [imageStats, setImageStats] = useState({
    authentic: 0,
    tampered: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
    
  const data = {
        labels: ['Tampered', 'Authentic'],
        datasets: [
            {
            label: '# of Images',
            data: [imageStats.tampered, imageStats.authentic],
            backgroundColor: [
                'rgba(255, 99, 132, 0.5)',
                'rgba(54, 162, 235, 0.5)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1.5)',
                'rgba(54, 162, 235, 1.5)',
            ],
            borderWidth: 1,
            },
        ],
    };
  useEffect(() => {
    const fetchData = async () => {
        setIsLoading(true);
        const response = await instance.get('/image-stats');
        setImageStats(response.data.message);
        setIsLoading(false);
    };
    fetchData();
  }, []);
  console.log(imageStats)
  return (
    <>
        {
            isLoading ?
            <Backdrop
                sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
                open={isLoading}
                >
                <p style={{ marginRight: '1rem', fontFamily: 'Montserrat', letterSpacing: '.2rem' }}>Loading</p><CircularProgress color="inherit" />
            </Backdrop>
            :
            <>
                <ResponsiveAppBar />
                <div style={{ backgroundColor: '#52B58812', paddingTop: '.5rem', paddingBottom: '17rem' }}>
                    <Container style = {{ display: 'flex' }}>
                        <Card style={{ padding: '1rem' }}>
                            <Pie options={{ plugins: { legend: {position: "bottom"} } }} data={data} />
                        </Card>
                    </Container>
                </div>
            </>
        }
    </>
  )
}

export default Analytics;