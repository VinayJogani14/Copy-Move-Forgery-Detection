import { Navigate } from "react-router";
import Home from "../pages/Home";
import GetStarted from "../pages/GetStarted";
import Result from "../pages/Result";
import Analytics from "../pages/Analytics";

const routes = [
  {
    path: "/",
    element: <Home />,
  },
  {
    path: "/get-started",
    element: <GetStarted />,
  },
  {
    path: "/result",
    element: <Result />,
  },
  {
    path: "/analytics",
    element: <Analytics />,
  },
];

export default routes;
