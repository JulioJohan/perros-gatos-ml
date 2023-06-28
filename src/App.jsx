import { useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';


const PerrosYGatos = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const otrocanvasRef = useRef(null);
  const resultadoRef = useRef(null);
  let currentStream = null;
  let facingMode = 'user';
  let modelo = null;
  const tamano = 400;

  useEffect(() => {
    const cargarModelo = async () => {
      console.log('Cargando modelo...');
      modelo = await tf.loadLayersModel('src/assets/model.json');
      console.log(modelo);
      console.log('Modelo cargado');
    };

    cargarModelo();
    mostrarCamara();

    return () => {
      // Detener la transmisión de video cuando se desmonte el componente
      if (currentStream) {
        currentStream.getTracks().forEach((track) => {
          track.stop();
        });
      }
    };
  }, []);

  const mostrarCamara = () => {
    const opciones = {
      audio: false,
      video: { width: tamano, height: tamano },
    };

    if (navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia(opciones)
        .then(function (stream) {
          currentStream = stream;
          videoRef.current.srcObject = currentStream;
          procesarCamara();
          predecir();
        })
        .catch(function (err) {
          alert('No se pudo utilizar la camara :(');
          console.log(err);
          alert(err);
        });
    } else {
      alert('No existe la funcion getUserMedia');
    }
  };

  const cambiarCamara = () => {
    if (currentStream) {
      currentStream.getTracks().forEach((track) => {
        track.stop();
      });
    }

    facingMode = facingMode === 'user' ? 'environment' : 'user';

    const opciones = {
      audio: false,
      video: { facingMode: facingMode, width: tamano, height: tamano },
    };

    navigator.mediaDevices
      .getUserMedia(opciones)
      .then(function (stream) {
        currentStream = stream;
        videoRef.current.srcObject = currentStream;
      })
      .catch(function (err) {
        console.log('Oops, hubo un error', err);
      });
  };

  const procesarCamara = () => {
    if (canvasRef.current && videoRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.drawImage(
        videoRef.current,
        0,
        0,
        tamano,
        tamano,
        0,
        0,
        tamano,
        tamano
      );
    }
    setTimeout(procesarCamara, 20);
  };

  const predecir = () => {
    if (modelo !== null) {
      resample_single(canvasRef.current, 100, 100, otrocanvasRef.current);

      // Hacer la predicción
      const ctx2 = otrocanvasRef.current.getContext('2d');
      const imgData = ctx2.getImageData(0, 0, 100, 100);

      let arr = [];
      let arr100 = [];

      for (let p = 0; p < imgData.data.length; p += 4) {
        const rojo = imgData.data[p] / 255;
        const verde = imgData.data[p + 1] / 255;
        const azul = imgData.data[p + 2] / 255;

        const gris = (rojo + verde + azul) / 3;

        arr100.push([gris]);
        if (arr100.length === 100) {
          arr.push(arr100);
          arr100 = [];
        }
      }

      arr = [arr];

      const tensor = tf.tensor4d(arr);
      const resultado = modelo.predict(tensor).dataSync();

      let respuesta;
      console.log(resultado)
      if (resultado <= 0.5) {
        respuesta = 'Gato';
      } else {
        respuesta = 'Perro';
      }
      resultadoRef.current.innerHTML = respuesta;
    }

    setTimeout(predecir, 150);
  };

  /**
   * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
   *
   * @param {HtmlElement} canvas
   * @param {int} width
   * @param {int} height
   * @param {boolean} resize_canvas if true, canvas will be resized. Optional.
   * Cambiado por RT, resize canvas ahora es donde se pone el chiqitillllllo
   */
  const resample_single = (canvas, width, height, resize_canvas) => {
    const width_source = canvas.width;
    const height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    const ratio_w = width_source / width;
    const ratio_h = height_source / height;
    const ratio_w_half = Math.ceil(ratio_w / 2);
    const ratio_h_half = Math.ceil(ratio_h / 2);

    const ctx = canvas.getContext('2d');
    const ctx2 = resize_canvas.getContext('2d');
    const img = ctx.getImageData(0, 0, width_source, height_source);
    const img2 = ctx2.createImageData(width, height);
    const data = img.data;
    const data2 = img2.data;

    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        const x2 = (i + j * width) * 4;
        let weight = 0;
        let weights = 0;
        let weights_alpha = 0;
        let gx_r = 0;
        let gx_g = 0;
        let gx_b = 0;
        let gx_a = 0;
        const center_y = (j + 0.5) * ratio_h;
        const yy_start = Math.floor(j * ratio_h);
        const yy_stop = Math.ceil((j + 1) * ratio_h);
        for (let yy = yy_start; yy < yy_stop; yy++) {
          const dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
          const center_x = (i + 0.5) * ratio_w;
          const w0 = dy * dy; //pre-calc part of w
          const xx_start = Math.floor(i * ratio_w);
          const xx_stop = Math.ceil((i + 1) * ratio_w);
          for (let xx = xx_start; xx < xx_stop; xx++) {
            const dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
            const w = Math.sqrt(w0 + dx * dx);
            if (w >= 1) {
              //pixel too far
              continue;
            }
            //hermite filter
            weight = 2 * w * w * w - 3 * w * w + 1;
            const pos_x = 4 * (xx + yy * width_source);
            //alpha
            gx_a += weight * data[pos_x + 3];
            weights_alpha += weight;
            //colors
            if (data[pos_x + 3] < 255) weight = weight * data[pos_x + 3] / 250;
            gx_r += weight * data[pos_x];
            gx_g += weight * data[pos_x + 1];
            gx_b += weight * data[pos_x + 2];
            weights += weight;
          }
        }
        data2[x2] = gx_r / weights;
        data2[x2 + 1] = gx_g / weights;
        data2[x2 + 2] = gx_b / weights;
        data2[x2 + 3] = gx_a / weights_alpha;
      }
    }

    ctx2.putImageData(img2, 0, 0);
  };

  return (
    <div>

        
        <main>
          <div className="px-4 py-2 my-2 text-center border-bottom">
            {/* <img class="d-block mx-auto mb-2" src="LogotipoV2-Simple.png" alt="" width="80" height="80"> */}
            <h1 className="display-5 fw-bold">Perros y gatos</h1>
            <div className="col-lg-6 mx-auto"></div>
          </div>

          <div className="b-example-divider"></div>

          <div className="container mt-5">
            <div className="row">
              <div className="col-12 col-md-4 offset-md-4 text-center">
                <video
                  ref={videoRef}
                  id="video"
                  playsInline
                  autoPlay
                  style={{ width: '1px' }}
                ></video>
                <button
                  className="btn btn-primary mb-2"
                  id="cambiar-camara"
                  onClick={cambiarCamara}
                >
                  Cambiar camara
                </button>
                <canvas
                    ref={canvasRef}
                    id="canvas"
                    width="400"
                    height="400"
                    style={{ maxWidth: '100%' }}
                ></canvas>
                <canvas
                 ref={otrocanvasRef}
                 id="otrocanvas"
                 width="150"
                 height="150"
                 style={{ display: 'none' }}
                ></canvas>
        <div ref={resultadoRef} id="resultado"></div>
              </div>
            </div>
          </div>

          <div className="b-example-divider"></div>

          <div className="b-example-divider mb-0"></div>
        </main>

        <script
          src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"
          integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM"
          crossOrigin="anonymous"
        ></script>

        <script
          src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.9.0/dist/tf.js"
          crossOrigin="anonymous"
        ></script>
    </div>

  );
};

export default PerrosYGatos;
