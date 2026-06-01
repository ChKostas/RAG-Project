import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '20s', target: 1 },
    { duration: '20s', target: 2 },
    { duration: '15s', target: 3 },
    { duration: '25s', target: 1 },
    { duration: '10s', target: 0 },
  ],
};

const questions = [
  "What is a premium?",
  "What is a deductible?"
];

export default function () {
  const question = questions[Math.floor(Math.random() * questions.length)];

  const payload = JSON.stringify({
    question: question
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '180s',
  };

  const res = http.post('http://host.docker.internal:7860/ask', payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(8);
}