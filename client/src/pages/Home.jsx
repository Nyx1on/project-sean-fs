import { useState } from "react";
import github from "../assets/github.svg";

const Home = () => {
  const [inputR, setInputR] = useState("");
  const [data, setData] = useState([]);

  async function getPredictions(inputReview) {
    if (inputR) {
      try {
        const response = await fetch(
          `http://localhost:8000/predict/?input_review=${inputReview}`
        );
        const res = await response.json();
        setData(res);
        console.log(res);
      } catch (error) {
        console.error(error);
      }
    } else {
      alert("The prompt field cannot be empty");
    }
  }
  const handleChange = (e) => {
    setInputR(e.target.value);
    console.log(setInputR);
  };
  return (
    <>
      <div className="home flex flex-row align-middle justify-between">
        <div className="left w-1/2">
          <h1 className="mt-32 font-fira font-extrabold text-[#ffffff] text-center text-[32px]">
            Sentiment Analysis
          </h1>
          <p className="mt-5 max-sm:m-5 text-[#2f9393] text-[24px] w-3/4 mx-auto">
            Instantly evaluate the sentiment of any comment by entering it into
            the prompt. Gain valuable insights into the emotional tone of your
            text with the model developed by us.
          </p>
          <p className="mt-5 max-sm:m-5 text-[#666e75] text-[14px]  w-3/4 mx-auto">
            <i>
              Note:For accurate results, please provide precise and detailed
              comments when using our website.
            </i>
          </p>
          <div className="flex flex-col">
            <input
              className="mx-auto mt-5 bg-white border bg-opacity-10 italic border-[#6A6D56] text-white text-xl rounded-lg focus:ring-[#2f9393] focus:border-[#2f9393] outline-none block w-3/4 max-sm:w-3/4 p-3"
              type="text"
              onChange={handleChange}
              name="input_review"
              placeholder="Enter the prompt"
              required
            />
            <br />
            <button
              type="button"
              className="border border-[#6A6D56] text-[18px] text-white bg-transparent hover:bg-[#46392A] hover:border-[#6A6D56] py-3 px-6 rounded-lg mx-auto transition-colors duration-300 ease-in-out"
              onClick={() => {
                getPredictions(inputR);
              }}
            >
              Send
            </button>
            <div className="mx-auto mt-16 mb-0">
              {data.Predictions === "positive" ? (
                <div className="text-[#888B64] text-[24px]">
                  The comment seems to be <strong>Positive</strong>
                </div>
              ) : (
                <></>
              )}
              {data.Predictions === "negative" ? (
                <div className="text-[#867659] text-[24px]">
                  The comment seems to be <strong>Negative</strong>
                </div>
              ) : (
                <></>
              )}
            </div>
            <div className="max-sm:mx-9 font-normal text-white absolute bottom-0 left-0 mb-4 ml-8 flex flex-row items-center">
              Follow us on Github:
              <a href="https://github.com/Nyx1on/project-sean-fs">
                <img src={github} alt="github icon" className="mx-2" />
              </a>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;
